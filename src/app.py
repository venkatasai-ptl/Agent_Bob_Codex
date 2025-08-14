from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import glob
from transcribe import transcribe_audio
from llm import get_llm_response
from flask_socketio import SocketIO, emit

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__, template_folder='../templates')
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

import threading
from flask import Response
import time
import json
import queue

# SSE setup
class MessageAnnouncer:
    def __init__(self):
        self.listeners = []

    def listen(self):
        self.listeners.append(queue.Queue(maxsize=5))
        return self.listeners[-1]

    def announce(self, msg):
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]

announcer = MessageAnnouncer()

def format_sse(data: str, event=None) -> str:
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg

@app.route('/listen', methods=['GET'])
def listen():
    def stream():
        messages = announcer.listen()
        while True:
            msg = messages.get()
            yield msg

    return Response(stream(), mimetype='text/event-stream')

@app.route('/simulate-capture', methods=['POST'])
def simulate_capture():
    """Endpoint to simulate audio capture and processing"""
    # Create a unique ID and timestamp for the simulation
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simulate processing with sample text
    sample_text = "This is a simulation of audio capture and processing."
    
    # Get streaming response
    def generate():
        for token in get_llm_response(sample_text, stream=True):
            # Send token via SSE
            announcer.announce(format_sse(json.dumps({"token": token}), "token"))
            yield token
    
    # Return streaming response
    return Response(generate(), mimetype='text/plain')

@app.route('/process', methods=['POST'])
def process_audio():
    # Create directories if they don't exist
    os.makedirs('data/recordings', exist_ok=True)
    os.makedirs('data/transcripts', exist_ok=True)
    os.makedirs('data/responses', exist_ok=True)
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save recording
    recording_filename = f"data/recordings/{timestamp}_{unique_id}.wav"
    audio_file.save(recording_filename)
    
    try:
        # Transcribe audio
        text = transcribe_audio(recording_filename)
        
        # Save transcript
        transcript_filename = f"data/transcripts/{timestamp}_{unique_id}.txt"
        with open(transcript_filename, 'w') as f:
            f.write(text)
        
        # Clear previous output in UI
        socketio.emit('clear')
        
        # Use streaming for LLM response
        def generate():
            response_buffer = []
            response_filename = f"data/responses/{timestamp}_{unique_id}.txt"
            
            # Open response file for writing
            with open(response_filename, 'w') as f:
                # Stream tokens from LLM
                for token in get_llm_response(text, stream=True):
                    if token:
                        response_buffer.append(token)
                        f.write(token)
                        f.flush()  # Ensure each token is written immediately
                        
                        # Send token to WebSocket clients
                        socketio.emit('token', {'token': token})
                        yield token
            
            # Save the full response at the end as well
            full_response = ''.join(response_buffer)
            with open(response_filename, 'w') as f:
                f.write(full_response)
            
            # Announce completion
            socketio.emit('complete')
        
        # Return streaming response
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_responses')
def get_responses():
    """Get all LLM responses for display"""
    responses = []
    for file in sorted(glob.glob('data/responses/*.txt'), reverse=True):
        try:
            with open(file, 'r') as f:
                text = f.read()
                # Extract timestamp from filename (format: YYYYMMDD_HHMMSS)
                filename = os.path.basename(file)
                timestamp_str = filename.split('_')[0]
                # Format timestamp for display
                timestamp = f"{timestamp_str[0:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]} {timestamp_str[9:11]}:{timestamp_str[11:13]}:{timestamp_str[13:15]}"
                responses.append({
                    "timestamp": timestamp,
                    "text": text
                })
        except:
            continue
    return jsonify(responses)

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to WebSocket'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
