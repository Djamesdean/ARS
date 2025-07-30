import vosk
import json
import numpy as np
import pyaudio
from vad import SileroVAD  
from noise_detection import calculate_energy


class VoskKeywordSpotting:
    """
    Vosk-based keyword spotting for wake word detection
    """
    def __init__(self, model_path="models/vosk-model-small-en-us-0.15"):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        print("✅ Vosk model loaded successfully")

    def process_audio(self, audio_bytes):
        """
        Process audio bytes and return transcription result
        Returns: (is_complete, transcribed_text)
        """
        try:
            if self.recognizer.AcceptWaveform(audio_bytes):
                # Complete sentence detected
                result = self.recognizer.Result()
                result_data = json.loads(result)
                text = result_data.get("text", "")
                return True, text
            else:
                # Partial word/phrase
                partial = self.recognizer.PartialResult()
                partial_data = json.loads(partial)
                text = partial_data.get("partial", "")
                return False, text
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            return False, ""


def run_vosk_pipeline():
    """
    Main function to run wake word detection using Vosk
    """
    print("Starting wake word detection system...")
    
    # Initialize components
    vosk_detector = VoskKeywordSpotting("models/vosk-model-small-en-us-0.15")
    silero_vad = SileroVAD()
    
    # Setup audio stream
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8000
    )
    
    stream.start_stream()
    print("🎤 Listening for 'hello'... Press Ctrl+C to stop")
    
    # Detection variables
    wake_word_count = 0
    silence_count = 0
    
    try:
        while True:
            # Capture audio chunk
            audio_data = stream.read(4000)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check if there's meaningful audio
            energy = calculate_energy(audio_array)
            voice_probability = silero_vad.detect_voice(audio_array)
            
            # Process speech or loud audio
            if voice_probability > 0.3 or energy > 100:
                silence_count = 0
                
                # Get transcription from Vosk
                is_complete, transcription = vosk_detector.process_audio(audio_data)
                
                if transcription:
                    print(f"Heard: '{transcription}'")
                    
                    # Check for wake word
                    if "hello" in transcription.lower():
                        wake_word_count += 1
                        print(f"🟢 ACCESS GRANTED - Wake word detected! (#{wake_word_count})")
                        print("-" * 40)
                    else:
                        print("🔴 ACCESS DENIED - Wake word not detected")
                        
            else:
                silence_count += 1
                
                # Reset after long silence (optional cleanup)
                if silence_count > 200:  # About 20 seconds
                    silence_count = 0
                    
    except KeyboardInterrupt:
        print(f"\nStopped. Wake words detected: {wake_word_count}")
    
    finally:
        # Cleanup audio resources
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    run_vosk_pipeline()