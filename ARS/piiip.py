import vosk
import json
import numpy as np

class VoskKeywordSpotting:
    def __init__(self, model_path="models/vosk-model-small-en-us-0.15"):
        # Initialize the Vosk model correctly
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

# Example usage:
if __name__ == "__main__":
    vosk_recognizer = VoskKeywordSpotting("models/vosk-model-small-en-us-0.15")  # Replace with your Vosk model path
    audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)  # Simulated audio data for testing
    transcription = vosk_recognizer.process_audio(audio_data)
    print(f"Detected Text: {transcription}")
