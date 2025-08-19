from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import glob
import base64
from transcribe import transcribe_audio
from llm import get_llm_response
from flask_socketio import SocketIO, emit
import json
from buffer_manager import BufferManager


load_dotenv()  # Load environment variables from .env file

app = Flask(__name__, template_folder='../templates')
# Generate a random secret key for session management
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret") 
socketio = SocketIO(app, cors_allowed_origins="*")
buffers = {}

# ---------------------------
# Helpers (deduped utilities)
# ---------------------------

def make_ids():
    """Return (unique_id_hex, now_datetime)."""
    return uuid.uuid4().hex, datetime.now()

def ts_slug(dt: datetime) -> str:
    """Datetime -> 'YYYYMMDD_HHMMSS'."""
    return dt.strftime("%Y%m%d_%H%M%S")

def human_ts_from_slug(slug: str) -> str:
    """
    'YYYYMMDD_HHMMSS' -> 'YYYY-MM-DD HH:MM:SS'
    Safe for filenames like '20250814_123456_<uuid>.txt' after extracting slug.
    """
    return f"{slug[0:4]}-{slug[4:6]}-{slug[6:8]} {slug[9:11]}:{slug[11:13]}:{slug[13:15]}"

def ensure_dirs():
    os.makedirs('data/recordings', exist_ok=True)
    os.makedirs('data/transcripts', exist_ok=True)
    os.makedirs('data/responses', exist_ok=True)
    os.makedirs('data/sessions', exist_ok=True)


def process_recording(recording_filename: str, session_id: str, slug: str | None = None, unique_id: str | None = None):
    """Transcribe audio, stream LLM tokens and save artifacts."""
    ensure_dirs()
    if slug is None or unique_id is None:
        unique_id, now = make_ids()
        slug = ts_slug(now)

    text = transcribe_audio(recording_filename)

    transcript_filename = f"data/transcripts/{slug}_{unique_id}.txt"
    with open(transcript_filename, 'w', encoding='utf-8') as f:
        f.write(text)

    socketio.emit('clear')

    response_filename = f"data/responses/{slug}_{unique_id}.txt"

    messages_text = build_messages_text(session_id, text, max_turns=5)

    prompt_filename = f"data/prompts/{slug}_{unique_id}.json"
    os.makedirs("data/prompts", exist_ok=True)
    with open(prompt_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": human_ts_from_slug(slug),
            "session_id": session_id,
            "transcript": text,
            "prompt": messages_text
        }, f, ensure_ascii=False, indent=2)

    response_buffer = []
    for token in stream_llm_tokens(messages_text):
        response_buffer.append(token)
        socketio.emit('token', {'token': token})

    full_response = ''.join(response_buffer)
    with open(response_filename, 'w', encoding='utf-8') as f:
        f.write(full_response)

    append_chat_history(
        session_id,
        human_ts_from_slug(slug),
        text,
        full_response
    )

    socketio.emit('complete')

    return {
        "status": "success",
        "recording_file": recording_filename,
        "transcript_file": transcript_filename,
        "response_file": response_filename,
        "timestamp": human_ts_from_slug(slug)
    }


def handle_buffer_complete(session_id: str, wav_path: str) -> None:
    """Callback from BufferManager when a segment is ready."""
    unique_id, now = make_ids()
    slug = ts_slug(now)
    recording_filename = f"data/recordings/{slug}_{unique_id}.wav"
    os.makedirs('data/recordings', exist_ok=True)
    os.replace(wav_path, recording_filename)
    try:
        process_recording(recording_filename, session_id, slug, unique_id)
    except Exception as e:
        print(f"Error processing buffered audio: {e}")

def stream_llm_tokens(text):
    """
    Standardized iterator over LLM tokens.
    """
    for token in get_llm_response(text, stream=True):
        if token:
            yield token

def get_session_id():
    """
    Return current session id from (in order):
      1) X-Session-Id header
      2) JSON body 'session_id'
      3) form field 'session_id'
      4) Flask session cookie
    Raise if none found.
    """
    # 1) Header
    sid = request.headers.get('X-Session-Id')
    if sid:
        return sid

    # 2) JSON body
    if request.is_json:
        data = request.get_json(silent=True) or {}
        sid = data.get('session_id')
        if sid:
            return sid

    # 3) Form field (multipart or urlencoded)
    sid = request.form.get('session_id')
    if sid:
        return sid

    # 4) Cookie (fallback)
    sid = session.get('session_id')
    if sid:
        return sid

    raise RuntimeError("No active session. Provide session_id or call /start-session first.")

def get_chat_file(session_id: str) -> str:
    """Ensure session dir exists and return path to its chat.json file."""
    session_dir = f"data/sessions/{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    return os.path.join(session_dir, "chat.json")

def append_chat_history(session_id: str, timestamp: str, user_text: str, assistant_text: str):
    """
    Append a single turn to the session's chat history JSON.
    Record shape: { "timestamp": ..., "user": ..., "assistant": ... }
    """
    chat_file = get_chat_file(session_id)

    # Load existing history if present
    history = []
    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f) or []
        except Exception:
            history = []

    # Append new turn
    history.append({
        "timestamp": timestamp,
        "user": user_text,
        "assistant": assistant_text
    })

    # Save back
    with open(chat_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def build_messages_text(session_id: str, transcript: str, max_turns: int = 5) -> str:
    """
    Assemble the full user message with:
      - [CONTEXT] resume + job description (from session files)
      - [CHAT_HISTORY] last N turns from chat.json
      - [NEW_PROMPT] the fresh transcript
      - [OUTPUT_INSTRUCTIONS] interview style guardrails
    """
    session_dir = f"data/sessions/{session_id}"
    resume_path = os.path.join(session_dir, "resume.txt")
    jd_path = os.path.join(session_dir, "job_description.txt")
    chat_file = get_chat_file(session_id)

    # Load resume/JD
    resume = ""
    if os.path.exists(resume_path):
        with open(resume_path, encoding="utf-8") as f:
            resume = f.read()

    jd = ""
    if os.path.exists(jd_path):
        with open(jd_path, encoding="utf-8") as f:
            jd = f.read()

    # Load last N turns of chat history
    history = []
    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f) or []
        except Exception:
            history = []

    # Chronological last N turns
    history_sorted = sorted(history, key=lambda t: t.get("timestamp", ""))[-max_turns:]
    hist_lines = []
    for t in history_sorted:
        q = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if q or a:
            hist_lines.append(f"- Q: {q}\n  A: {a}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(none)"

    return f"""[CONTEXT]
RESUME:
{resume}

JOB_DESCRIPTION:
{jd}

[CHAT_HISTORY_LAST_{max_turns}_TURNS]
{hist_block}

[NEW_PROMPT]
You are answering the interviewer’s last question based on the transcript below.

TRANSCRIPT:
{transcript}

[OUTPUT_INSTRUCTIONS]
- Speak as me, in first person.
- Be concise and confident. No fluff or hedging. No invented facts (no fake names/dates/employers).
- Use STAR when helpful; quantify impact (metrics, scale, tools) if available.
- If details are missing, keep them generic but realistic.
- End with one-line takeaway.

Return a clean answer. If STAR doesn’t fully apply, answer plainly without inventing details.
"""



# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_audio():
    """
    Accepts an audio file, transcribes it, streams model tokens via WebSocket,
    and writes a single final response file at the end (no duplicate writes).
    Returns JSON when complete. No HTTP streaming body is used—WebSockets only.
    """
    ensure_dirs()

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    # Require an active session; do not default to 'default'
    try:
        current_session_id = get_session_id()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    print(f"Using session ID for chat history: {current_session_id}")

    audio_file = request.files['audio']
    unique_id, now = make_ids()
    slug = ts_slug(now)
    recording_filename = f"data/recordings/{slug}_{unique_id}.wav"
    audio_file.save(recording_filename)
    try:
        result = process_recording(recording_filename, current_session_id, slug, unique_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/start-session', methods=['POST'])
def start_session():
    """
    Start a new session with a resume and job description.
    """
    data = request.get_json()

    # Validate input
    if not data or 'resume' not in data or 'job_description' not in data:
        return jsonify({"error": "Missing resume or job description"}), 400

    # Create session directory structure
    session_id = uuid.uuid4().hex
    session_dir = f"data/sessions/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    # Save resume and job description to files
    with open(f"{session_dir}/resume.txt", 'w', encoding='utf-8') as f:
        f.write(data['resume'])

    with open(f"{session_dir}/job_description.txt", 'w', encoding='utf-8') as f:
        f.write(data['job_description'])

    # Store session ID in Flask session
    session['session_id'] = session_id

    # Also persist the "current" session id for non-cookie clients
    ensure_dirs()
    with open('data/last_session_id.txt', 'w', encoding='utf-8') as f:
        f.write(session_id)

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "message": "Session started successfully"
    })

@app.route('/active-session', methods=['GET'])
def active_session():
    """
    Return the latest known session_id for non-cookie clients.
    Prefers the cookie session if available, else falls back to data/last_session_id.txt.
    """
    # 1) try cookie-backed Flask session
    sid = session.get('session_id')
    if sid:
        return jsonify({"session_id": sid})

    # 2) fallback to last written session id
    try:
        with open('data/last_session_id.txt', 'r', encoding='utf-8') as f:
            sid = f.read().strip()
    except Exception:
        sid = None

    if not sid:
        return jsonify({"error": "No active session"}), 404

    return jsonify({"session_id": sid})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    try:
        session_id = get_session_id()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    
    chat_file = get_chat_file(session_id)

    history = []
    if os.path.exists(chat_file):
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f) or []
        except Exception:
            history = []

    # newest first (optional)
    history.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    return jsonify(history)


# ---------------------------
# Socket.IO events
# ---------------------------

@socketio.on('audio_frame')
def handle_audio_frame(data):
    session_id = data.get('session_id')
    frame_b64 = data.get('frame')
    if not session_id or not frame_b64:
        return
    frame = base64.b64decode(frame_b64)
    bm = buffers.get(session_id)
    if bm is None:
        bm = BufferManager(session_id, handle_buffer_complete)
        buffers[session_id] = bm
    bm.add_frame(frame)


@socketio.on('audio_stop')
def handle_audio_stop(data):
    session_id = data.get('session_id')
    bm = buffers.get(session_id)
    if bm:
        bm.flush()

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to WebSocket'})

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == '__main__':
    # debug=True is fine for development, but make sure to configure properly for production.
    socketio.run(app, debug=True, use_reloader=False)
