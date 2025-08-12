import subprocess
import sys
import os
import signal
import time

def run_flask():
    """Start the Flask server"""
    subprocess.run([sys.executable, "app.py"])

def run_audio_capture():
    """Start the audio capture"""
    subprocess.run([sys.executable, "audio_capture.py"])

if __name__ == "__main__":
    # Start Flask in a subprocess
    flask_process = subprocess.Popen([sys.executable, "src/app.py"])
    
    # Start audio capture in a subprocess
    audio_process = subprocess.Popen([sys.executable, "src/audio_capture.py"])
    
    try:
        print("Both processes started. Press Ctrl+C to stop.")
        # Wait indefinitely until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping processes...")
        # Terminate both processes
        flask_process.terminate()
        audio_process.terminate()
        # Wait for them to finish
        flask_process.wait()
        audio_process.wait()
        print("Processes stopped.")
