import pyaudio
import wave
import webrtcvad
import time
import requests
import os
from datetime import datetime
import uuid

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION = 30  # ms
CHUNK_SIZE = int(RATE * CHUNK_DURATION / 1000)
VAD_AGGRESSIVENESS = 3
SILENCE_TIMEOUT = 2  # seconds of silence to consider speech ended

def find_loopback_device(p):
    """Find stereo mix device index for system audio capture"""
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if 'stereo mix' in dev_info['name'].lower() and dev_info['maxInputChannels'] > 0:
            return i
    return None

def capture_audio_segment():
    """Capture audio segments with VAD and send to backend"""
    p = pyaudio.PyAudio()
    device_index = find_loopback_device(p)
    
    if device_index is None:
        print("No loopback device found. Please ensure you have Stereo Mix enabled.")
        return
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK_SIZE)
    
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frames = []
    speech_detected = False
    silence_count = 0
    start_time = time.time()
    
    print("Listening for system audio...")
    
    try:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            is_speech = vad.is_speech(chunk, RATE)
            
            if is_speech:
                frames.append(chunk)
                speech_detected = True
                silence_count = 0
            elif speech_detected:
                frames.append(chunk)
                silence_count += 1
                
                # If we've had enough silence, consider speech ended
                if silence_count * CHUNK_DURATION / 1000 >= SILENCE_TIMEOUT:
                    # Process the audio segment
                    process_audio_segment(b''.join(frames))
                    frames = []
                    speech_detected = False
                    silence_count = 0
    except KeyboardInterrupt:
        print("Stopping capture")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_audio_segment(audio_data):
    """Send audio segment to backend for processing"""
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{unique_id}.wav"
    
    # Save temporary audio file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    
    # Send to backend
    try:
        with open(filename, 'rb') as f:
            response = requests.post(
                "http://localhost:5000/process",
                files={'audio': (filename, f, 'audio/wav')}
            )
        if response.status_code == 200:
            print(f"Processed audio segment: {response.json()['response']}")
        else:
            print(f"Error processing audio: {response.text}")
    except Exception as e:
        print(f"Error sending audio to backend: {str(e)}")
    finally:
        # Clean up temporary file
        os.remove(filename)

if __name__ == "__main__":
    capture_audio_segment()
