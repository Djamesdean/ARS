
import time
import logging
import numpy as np
from config import *
from audio_capture import AudioCapture
from vad import SileroVAD
from noise_detection import noise_filter, calculate_energy
from picovoice_detection import detect_wake_word
from vosk_detection import run_vosk_pipeline
from whisper_detection import test_whisper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioResponseSystem:
    def __init__(self):
       
        self.audio_capture = AudioCapture(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            channels=CHANNELS
        )
        
        # Initialize Silero VAD
        try:
            self.silero_vad = SileroVAD()
            logger.info("Silero VAD initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            logger.error("Please check your internet connection for first-time model download")
            raise
        
        self.noise_profile = None
        
    def calibrate_noise(self, samples: int = 30) -> None:
        
        logger.info(f"Calibrating noise profile with {samples} samples...")
        logger.info("Please stay quiet during calibration...")
        
        noise_data = []
        
        for i in range(samples):
            audio_data = self.audio_capture.capture_audio()
            if len(audio_data) > 0:
                noise_data.append(audio_data)
            time.sleep(0.1)
        
        if noise_data:
            # Calculate average noise profile
            self.noise_profile = np.mean(noise_data, axis=0)
            avg_energy = np.mean([calculate_energy(data) for data in noise_data])
            logger.info(f"Noise calibration complete. Average quiet energy: {avg_energy:.2f}")
        else:
            logger.warning("No audio data captured during calibration")
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> dict:

        if len(audio_data) == 0:
            return {
                'voice_detected': False,
                'voice_probability': 0.0,
                'energy_level': 0.0,
                'timestamp': time.time()
            }
        
        # Calculate energy level first 
        energy = calculate_energy(audio_data)
        
        # Skip processing if energy is too low 
        if energy < ENERGY_THRESHOLD:
            return {
                'voice_detected': False,
                'voice_probability': 0.0,
                'energy_level': energy,
                'timestamp': time.time()
            }
        
        # Apply noise filtering
        filtered_audio = noise_filter(audio_data, SAMPLE_RATE)
        
        # Use Silero VAD for voice detection
        voice_prob = self.silero_vad.detect_voice(filtered_audio)
        voice_detected = voice_prob > VOICE_THRESHOLD
        
        return {
            'voice_detected': voice_detected,
            'voice_probability': voice_prob,
            'energy_level': energy,
            'timestamp': time.time()
        }
    
    def run_continuous(self) -> None:
        """Run continuous audio processing."""
        logger.info("Starting continuous audio processing...")
        
        # Calibrate noise profile
        self.calibrate_noise()
        
        # Start continuous capture
        self.audio_capture.start_continuous_capture()
        
        try:
            logger.info("Listening for voice... (Press Ctrl+C to stop)")
            while True:
                audio_data = self.audio_capture.get_audio_data()
                if audio_data is not None:
                    results = self.process_audio_chunk(audio_data)
                    
                    if results['voice_detected']:
                        logger.info(f"🎤 Voice detected! "
                                  f"Probability: {results['voice_probability']:.2f}, "
                                  f"Energy: {results['energy_level']:.0f}")
                 
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            logger.info("Stopping audio processing...")
        finally:
            self.audio_capture.stop_capture()
    
    def run_single_shot(self) -> None:
        """Run single-shot audio processing."""
        logger.info("Starting single-shot audio processing...")
        
        with self.audio_capture:
            try:
                logger.info("Listening for voice... (Press Ctrl+C to stop)")
                while True:
                    audio_data = self.audio_capture.capture_audio()
                    if len(audio_data) > 0:
                        results = self.process_audio_chunk(audio_data)
                        
                        if results['voice_detected']:
                            logger.info(f"🎤 Voice detected! "
                                      f"Probability: {results['voice_probability']:.2f}, "
                                      f"Energy: {results['energy_level']:.0f}")
                        elif results['energy_level'] > ENERGY_THRESHOLD:
                            # Log non-voice audio activity
                            logger.debug(f"Audio detected (not voice): "
                                       f"Probability: {results['voice_probability']:.2f}, "
                                       f"Energy: {results['energy_level']:.0f}")
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                logger.info("Stopping audio processing...")

    def test_model(self) -> None:
        
        logger.info("Testing Silero VAD model...")
        
        try:
            # Capture a single chunk
            audio_data = self.audio_capture.capture_audio()
            
            if len(audio_data) > 0:
                results = self.process_audio_chunk(audio_data)
                logger.info(f"Test results: Voice={results['voice_detected']}, "
                          f"Probability={results['voice_probability']:.2f}, "
                          f"Energy={results['energy_level']:.0f}")
                logger.info("Model test successful!")
            else:
                logger.warning("No audio data captured")
                
        except Exception as e:
            logger.error(f"Model test failed: {e}")

def main():

    
    ars = AudioResponseSystem()
        
    print("\nProvide your Input")
    print("=" * 40)
    print("1. Continuous processing ")
    print("2. Single-shot processing")
    print("3. Test Speech detectioon model")
    print("4. Test Picovoice Wake Word Detection")
    print("5. Test Vosk wake Word Detection")
    print("6. Test Whisper API")
    print("=" * 40)
        
    choice = input("Choose mode (1-6): ").strip()
        
    if choice == "1":
        ars.run_continuous()
    elif choice == "2":
        ars.run_single_shot()
    elif choice == "3":
        ars.test_model()
    elif choice == "4":
        detect_wake_word()
    elif choice == "5":
        run_vosk_pipeline()
    elif choice == "6":
        test_whisper()
            

if __name__ == "__main__":
    main()