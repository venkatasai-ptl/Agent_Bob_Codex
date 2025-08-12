from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import glob
from transcribe import transcribe_audio
from llm import get_llm_response

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    return render_template('index.html')

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
        
        # Get LLM response
        llm_response = get_llm_response(text)
        
        # Save LLM response
        response_filename = f"data/responses/{timestamp}_{unique_id}.txt"
        with open(response_filename, 'w') as f:
            f.write(llm_response)
        
        # Return the response to frontend
        return jsonify({"response": llm_response})
    
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

if __name__ == '__main__':
    app.run(debug=True)
