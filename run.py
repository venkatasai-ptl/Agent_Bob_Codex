import subprocess
import sys

if __name__ == "__main__":
    # Start Flask in a subprocess
    flask_process = subprocess.Popen([sys.executable, "src/app.py"])

    try:
        print("Flask server started. Press Ctrl+C to stop.")
        flask_process.wait()
    except KeyboardInterrupt:
        print("\nStopping Flask server...")
        flask_process.terminate()
        flask_process.wait()
        print("Flask server stopped.")
