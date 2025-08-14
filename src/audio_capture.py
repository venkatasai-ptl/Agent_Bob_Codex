import pyaudiowpatch as pyaudio
import wave
import webrtcvad
import time
import requests
import os
from datetime import datetime
import uuid
import audioop  # stdlib resampling

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION = 30  # ms
CHUNK_SIZE = int(RATE * CHUNK_DURATION / 1000)
VAD_AGGRESSIVENESS = 3
SILENCE_TIMEOUT = 2  # seconds of silence to consider speech ended

def find_loopback_device(p):
    """Return the WASAPI default OUTPUT device index for loopback capture."""
    for h in range(p.get_host_api_count()):
        hai = p.get_host_api_info_by_index(h)
        if 'wasapi' in hai['name'].lower():
            idx = hai.get('defaultOutputDevice', -1)
            return idx if idx != -1 else None
    return None



def capture_audio_segment():
    """Capture audio segments with VAD and send to backend"""
    p = pyaudio.PyAudio()
    device_index = find_loopback_device(p)
    
    # Fallback to Stereo Mix device if no WASAPI loopback found
    if device_index is None:
        print("No WASAPI loopback device found. Using Stereo Mix as fallback.")
        # Try to find Stereo Mix device
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                if 'stereo mix' in dev_info['name'].lower() and dev_info['maxInputChannels'] > 0:
                    device_index = i
                    print(f"Using Stereo Mix device: {dev_info['name']}")
                    break
            except:
                continue
        
        if device_index is None:
            print("No Stereo Mix device found. Please check your audio settings.")
            print("\nAvailable audio devices:")
            for i in range(p.get_device_count()):
                try:
                    dev_info = p.get_device_info_by_index(i)
                    print(f"{i}: {dev_info['name']} (Input channels: {dev_info['maxInputChannels']}, Host API: {dev_info['hostApi']})")
                except Exception as e:
                    print(f"Error getting device info for index {i}: {str(e)}")
            return
    
    # Use default input device with standard parameters
    print("Using default input device with standard parameters")
    
    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    stream_channels = 1
    device_rate = RATE
    frames_per_buffer = CHUNK_SIZE
    
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frames = []
    speech_detected = False
    silence_count = 0
    start_time = time.time()
    
    print("Listening for system audio...")
    
    # Initialize resampler state and buffer
    resample_state = None
    vad_buf = bytearray()
    
    try:
        while True:
            chunk = stream.read(frames_per_buffer, exception_on_overflow=False)
            
            # Convert to mono based on stream's actual channel count
            if stream_channels == 2:
                mono_16le = audioop.tomono(chunk, 2, 0.5, 0.5)
            else:
                mono_16le = chunk
            
            # Resample to 16kHz with persistent state
            if device_rate != 16000:
                mono_16k, resample_state = audioop.ratecv(
                    mono_16le, 2, 1, device_rate, 16000, resample_state
                )
            else:
                mono_16k = mono_16le
            
            # Accumulate and slice into exact 30ms frames (960 bytes)
            vad_buf.extend(mono_16k)
            
            while len(vad_buf) >= 960:
                frame = bytes(vad_buf[:960])
                del vad_buf[:960]
                
                is_speech = vad.is_speech(frame, 16000)
                
                if is_speech:
                    frames.append(frame)  # Store mono@16k
                    speech_detected = True
                    silence_count = 0
                elif speech_detected:
                    frames.append(frame)
                    silence_count += 1
                    
                    # If we've had enough silence, consider speech ended
                    if silence_count * CHUNK_DURATION / 1000 >= SILENCE_TIMEOUT:
                        # Process the audio segment
                        process_audio_segment(b''.join(frames))
                        frames.clear()
                        speech_detected = False
                        silence_count = 0
    except KeyboardInterrupt:
        print("Stopping capture")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

import io

def process_audio_segment(audio_data):
    """Send audio segment to backend for processing using in-memory buffer"""
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{unique_id}.wav"
    
    try:
        # Create in-memory WAV file (mono@16kHz)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit samples
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(audio_data)  # Already processed
        
        # Reset buffer position to start
        wav_buffer.seek(0)
        
        # Send to backend
        response = requests.post(
            "http://localhost:5000/process",
            files={'audio': (filename, wav_buffer, 'audio/wav')}
        )
        
        if response.status_code == 200:
            print("Audio segment sent for processing")
        else:
            print(f"Error processing audio: {response.text}")
            
    except Exception as e:
        print(f"Error sending audio to backend: {str(e)}")
        
    finally:
        # Close buffer to free memory
        wav_buffer.close()

if __name__ == "__main__":
    capture_audio_segment()
