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
    """Find WASAPI loopback device for system audio capture"""
    # Prefer WASAPI loopback devices
    wasapi_devices = []
    
    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            # Look for WASAPI loopback devices
            if 'WASAPI' in dev_info['hostApiName'] and 'loopback' in dev_info['name'].lower():
                wasapi_devices.append((i, dev_info))
        except:
            continue
    
    # Prefer devices with stereo support
    for i, dev in wasapi_devices:
        if dev['maxInputChannels'] >= 2:
            return i
    
    # Fallback to first WASAPI device if no stereo
    if wasapi_devices:
        return wasapi_devices[0][0]
    
    # Final fallback: look for any loopback device
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if 'loopback' in dev_info['name'].lower() and dev_info['maxInputChannels'] > 0:
            return i
    
    return None

def convert_stereo_to_mono(data):
    """Convert stereo audio (2 channels) to mono"""
    # Each sample is 2 bytes (16-bit), stereo has 2 channels
    mono_data = bytearray()
    for i in range(0, len(data), 4):
        # Extract left and right channels (16-bit samples)
        left = int.from_bytes(data[i:i+2], 'little', signed=True)
        right = int.from_bytes(data[i+2:i+4], 'little', signed=True)
        # Average channels
        mono_sample = (left + right) // 2
        # Convert back to bytes
        mono_data.extend(mono_sample.to_bytes(2, 'little', signed=True))
    return bytes(mono_data)

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
    
    # Try to open WASAPI loopback
    try:
        stream = p.open(format=FORMAT,
                        channels=2,  # WASAPI loopback is always stereo
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE,
                        as_loopback=True)
        print("Using WASAPI loopback for system audio capture")
    except Exception as e:
        print(f"Error opening WASAPI loopback: {str(e)}")
        print("Falling back to standard device")
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
            # Convert to mono if needed (VAD requires mono audio)
            if CHANNELS == 2:
                mono_chunk = convert_stereo_to_mono(chunk)
            else:
                mono_chunk = chunk
            
            is_speech = vad.is_speech(mono_chunk, RATE)
            
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

import io

def process_audio_segment(audio_data):
    """Send audio segment to backend for processing using in-memory buffer"""
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{unique_id}.wav"
    
    try:
        # Create in-memory WAV file (always mono for Whisper)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Whisper requires mono
            wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            
            # Convert to mono if needed
            if CHANNELS == 2:
                mono_data = convert_stereo_to_mono(audio_data)
                wf.writeframes(mono_data)
            else:
                wf.writeframes(audio_data)
        
        # Reset buffer position to start
        wav_buffer.seek(0)
        
        # Send to backend
        response = requests.post(
            "http://localhost:5000/process",
            files={'audio': (filename, wav_buffer, 'audio/wav')}
        )
        
        if response.status_code == 200:
            print(f"Processed audio segment: {response.json()['response']}")
        else:
            print(f"Error processing audio: {response.text}")
            
    except Exception as e:
        print(f"Error sending audio to backend: {str(e)}")
        
    finally:
        # Close buffer to free memory
        wav_buffer.close()

if __name__ == "__main__":
    capture_audio_segment()
