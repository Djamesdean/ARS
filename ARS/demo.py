# demo_assistant.py

import time
import logging
import numpy as np
import pyaudio
import threading
import json
import os
from enum import Enum

# Internal modules
from config import SAMPLE_RATE
from vad import SileroVAD
from noise_detection import calculate_energy
from whisper_detection import capture_and_transcribe
import vosk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssistantState(Enum):
    LISTENING = 1
    RECORDING_COMMAND = 2
    PROCESSING_COMMAND = 3

class VoiceAssistant:
    def __init__(self):
        logger.info("Initializing Voice Assistant")

        # Init Silero VAD
        self.vad = SileroVAD()

        # Init Vosk
        model_path = "models/vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {model_path}")
        self.vosk_model = vosk.Model(model_path)
        self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)

        # Audio config
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=2048
        )

        self.state = AssistantState.LISTENING
        self.running = True
        self.noise_threshold = 100  # Default, will be updated in calibration

        self.calibrate_noise()

    def calibrate_noise(self, samples: int = 30):
        logger.info("🔧 Calibrating ambient noise level. Please stay quiet...")
        collected_energies = []
        for _ in range(samples):
            data = self.stream.read(2048, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16)
            collected_energies.append(calculate_energy(audio_np))
            time.sleep(0.1)

        avg_energy = np.mean(collected_energies)
        self.noise_threshold = max(100, avg_energy * 1.5)
        logger.info(f"✅ Noise calibration complete. Energy threshold set to {self.noise_threshold:.2f}")

    def listen_loop(self):
        logger.info("👂 Assistant is running. Say 'hello' to trigger command mode")
        try:
            while self.running:
                if self.state == AssistantState.LISTENING:
                    audio_data = self.stream.read(2048, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)

                    energy = calculate_energy(audio_np)
                    if energy < self.noise_threshold:
                        continue

                    voice_prob = self.vad.detect_voice(audio_np)
                    if voice_prob > 0.3:
                        print(f"🎤 Speech detected... energy={energy:.1f}, prob={voice_prob:.2f}", end='\r')

                        is_final = self.vosk_recognizer.AcceptWaveform(audio_data)

                        if is_final:
                            result = json.loads(self.vosk_recognizer.Result())
                            text = result.get("text", "").lower()
                            if "hello" in text:
                                print("\n🟢 ACCESS GRANTED")
                                threading.Thread(target=self.record_command, daemon=True).start()
                                self.state = AssistantState.RECORDING_COMMAND
                        else:
                            partial = json.loads(self.vosk_recognizer.PartialResult())
                            partial_text = partial.get("partial", "").lower()
                            if "hello" in partial_text:
                                print("\n🟢 ACCESS GRANTED")
                                threading.Thread(target=self.record_command, daemon=True).start()
                                self.state = AssistantState.RECORDING_COMMAND

                time.sleep(0.005)
        except KeyboardInterrupt:
            logger.info("🛑 Assistant stopped by user")
        finally:
            self.cleanup()

    def record_command(self):
        try:
            self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            time.sleep(1)
            logger.info("🎙️ Recording your command for 3 seconds... Speak now!")
            self.state = AssistantState.PROCESSING_COMMAND

            transcript = capture_and_transcribe(duration_seconds=3.0)
            if transcript:
                print("\n==================== TRANSCRIPTION ====================")
                print(f"📝 You said: {transcript}")
                print("======================================================")
            else:
                logger.warning("❌ No speech detected or transcription failed")
        except Exception as e:
            logger.error(f"❌ Error during command processing: {e}")
        finally:
            logger.info("👂 Listening again for 'hello'")
            self.state = AssistantState.LISTENING

    def cleanup(self):
        self.running = False
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()
        logger.info("🧹 Audio resources cleaned up")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.listen_loop()
