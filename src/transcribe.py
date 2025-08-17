import openai
import os

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using OpenAI Whisper
    Returns transcribed text
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    return transcript.text
