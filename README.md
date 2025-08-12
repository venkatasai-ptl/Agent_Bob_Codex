# Audio-to-LLM System

This application captures system audio, transcribes it using OpenAI Whisper, generates responses using OpenAI GPT, and displays them in a web interface.

## Features
- 🎙️ Real-time system audio capture
- 🔍 Voice Activity Detection for speech segmentation
- 📝 Speech-to-text with OpenAI Whisper
- 💬 Response generation with OpenAI GPT
- 🌐 Web interface for real-time display
- 💾 Automatic storage of recordings, transcripts, and responses

## Requirements
- Python 3.10+
- OpenAI API key
- Stereo Mix enabled (Windows)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/venkatasai-ptl/Agent_Bob.git
cd Agent_Bob
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Usage
Start the application:
```bash
python run.py
```

Access the web interface:  
http://localhost:5000

The system will automatically:
- Capture audio from your speakers
- Detect speech segments
- Transcribe and process through LLM
- Display responses in real-time

Press Ctrl+C to stop the application.

## Project Structure
```
Agent_Bob/
├── data/                 # Data storage
│   ├── recordings/       # Audio files
│   ├── responses/        # LLM responses
│   └── transcripts/      # Text transcripts
├── src/                  # Source code
│   ├── app.py            # Flask application
│   ├── audio_capture.py  # Audio recorder
│   ├── llm.py            # Response generator
│   └── transcribe.py     # Audio-to-text
├── templates/            # Web interface
│   └── index.html
├── .env                  # API key configuration
├── requirements.txt      # Dependencies
└── run.py                # Control script
```

## Configuration
- Set `OPENAI_API_KEY` in `.env`
- Modify `audio_capture.py` for audio device settings
- Adjust VAD sensitivity in `audio_capture.py` (VAD_AGGRESSIVENESS)

## Notes
- Requires "Stereo Mix" enabled in Windows sound settings
- All generated files are stored in `data/` for review
