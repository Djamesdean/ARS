import vosk
import json
import numpy as np
from vad import SileroVAD  # Assuming Silero VAD is in 'vad.py'
from noise_detection import noise_filter  # Optional noise filtering from 'noise_detection.py'
from audio_capture import AudioCapture
class VoskKeywordSpotting:
    def __init__(self, model_path="models/vosk-model-small-en-us-0.15"):
        # Initialize Vosk with the downloaded model path
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)  # Sample rate should match Silero VAD (16kHz)

    def process_audio(self, audio_data: np.ndarray) -> str:
        """
        Process a chunk of audio data with Vosk to detect wake words.
        Args:
            audio_data: The raw audio data in bytes.
        Returns:
            Detected transcription text.
        """
        # Vosk accepts audio data as raw bytes, so convert the numpy array to bytes
        if self.recognizer.AcceptWaveform(audio_data.tobytes()):
            result = self.recognizer.Result()  # Get result from Vosk recognizer
            result_json = json.loads(result)
            return result_json.get("text", "")
        else:
            return ""  # Return empty string if no valid result

def process_vosk_pipeline(audio_data: np.ndarray, silero_vad: SileroVAD, vosk_recognizer: VoskKeywordSpotting):
    # Step 1: Use Silero VAD to detect speech
    voice_prob = silero_vad.detect_voice(audio_data)
    
    if voice_prob > 0.5:  # If speech is detected (adjust threshold as needed)
        print("Speech detected!")
        
        # Step 2: Apply noise filtering (optional but recommended for cleaner audio)
        filtered_audio = noise_filter(audio_data, 16000)  # Adjust sample rate if needed
        
        # Step 3: Pass the detected speech to Vosk for wake word detection
        transcription = vosk_recognizer.process_audio(filtered_audio)
        print(f"Vosk transcription: {transcription}")
        
        # Step 4: Check if a specific wake word is detected (e.g., "hello")
        if "hello" in transcription.lower():
            print("Wake word 'hello' detected!")
            # Trigger further actions here (e.g., execute command)
        else:
            print("No wake word detected.")
    else:
        print("No speech detected.")

# Example function to run the Vosk pipeline:

def run_vosk_pipeline():
    # Initialize models
    silero_vad = SileroVAD()
    vosk_recognizer = VoskKeywordSpotting("models/vosk-model-small-en-us-0.15")  # Adjust this path as needed

    # ✅ Create instance of your audio capture class
    audio_capture = AudioCapture(sample_rate=16000, chunk_size=512, channels=1)

    print("🎙️ Listening for speech and wake word... Press Ctrl+C to stop.")

    try:
        while True:
            audio_data = audio_capture.capture_audio()
            if len(audio_data) > 0:
                process_vosk_pipeline(audio_data, silero_vad, vosk_recognizer)
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")

