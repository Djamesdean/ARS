# main_with_wakeword.py
import time
import logging
import numpy as np
from config import *
from audio_capture import AudioCapture
from vad import SileroVAD
from noise_detection import noise_filter, calculate_energy
from openwakeword1 import WakeWordDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioResponseSystem:
    def __init__(self):
        """Initialize the Audio Response System with Silero VAD and Wake Word Detection."""
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
        
        # Initialize Wake Word Detector
        try:
            self.wake_word_detector = WakeWordDetector(model_names=['alexa'], threshold=0.5)
            logger.info("Wake word detector initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wake word detector: {e}")
            self.wake_word_detector = None
        
        self.noise_profile = None
        self.wake_word_active = False
        
    def calibrate_noise(self, samples: int = 30) -> None:
        """Calibrate noise profile for better detection."""
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
        """
        Process a single audio chunk using Silero VAD only.
        Args:
            audio_data: Raw audio data
        Returns:
            dict: Processing results
        """
        if len(audio_data) == 0:
            return {
                'voice_detected': False,
                'voice_probability': 0.0,
                'energy_level': 0.0,
                'timestamp': time.time()
            }
        
        # Calculate energy level first (quick check)
        energy = calculate_energy(audio_data)
        
        # Skip processing if energy is too low (silence)
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
                        
                        # Here you can add additional processing like:
                        # - Save audio segment
                        # - Trigger speech recognition
                        # - Execute voice commands
                        # - Send to keyword spotting
                        
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            logger.info("Stopping audio processing...")
        finally:
            self.audio_capture.stop_capture()
    
    def run_wake_word_mode(self) -> None:
        """Run wake word detection mode."""
        if self.wake_word_detector is None:
            logger.error("Wake word detector not available")
            return
        
        logger.info("Starting wake word detection mode...")
        logger.info("Say 'Alexa' to activate the system...")
        
        try:
            # Start wake word detection
            self.wake_word_detector.start_listening()
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping wake word detection...")
        finally:
            if self.wake_word_detector:
                self.wake_word_detector.stop_listening()
    
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

    def test_wake_word_only(self) -> None:
        """Test wake word detection only."""
        if self.wake_word_detector is None:
            logger.error("Wake word detector not available")
            return
            
        logger.info("Testing wake word detection for 30 seconds...")
        logger.info("Try saying 'Alexa' multiple times...")
        
        # Simple test without VAD
        audio_capture = AudioCapture(sample_rate=16000, chunk_size=160, channels=1)
        
        try:
            start_time = time.time()
            while time.time() - start_time < 30:  # Test for 30 seconds
                audio_data = audio_capture.capture_audio()
                if len(audio_data) > 0:
                    energy = calculate_energy(audio_data)
                    
                    if energy > ENERGY_THRESHOLD:
                        # Apply noise filtering
                        filtered_audio = noise_filter(audio_data, SAMPLE_RATE)
                        
                        # Test wake word detection
                        predictions = self.wake_word_detector.detect_wake_word(filtered_audio)
                        
                        if predictions:
                            max_score = max(predictions.values())
                            if max_score > 0.1:  # Lower threshold for testing
                                logger.info(f"Wake word predictions: {predictions}")
                                
                            if max_score > 0.5:  # Actual detection threshold
                                best_model = max(predictions.items(), key=lambda x: x[1])
                                logger.info(f"🎯 Wake word detected: {best_model[0]} (confidence: {best_model[1]:.3f})")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Test interrupted")
        finally:
            audio_capture.stop_capture()
            
        logger.info("Wake word test completed")

    def test_model(self) -> None:
        """Test the Silero VAD model with a single audio capture."""
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
    """Main function to run the Audio Response System."""
    try:
        ars = AudioResponseSystem()
        
        print("\nAudio Response System - Enhanced with Wake Word Detection")
        print("=" * 60)
        print("1. Continuous VAD processing (recommended)")
        print("2. Single-shot VAD processing")
        print("3. Wake word detection mode")
        print("4. Test wake word detection only")
        print("5. Test VAD model")
        print("=" * 60)
        
        choice = input("Choose mode (1-5): ").strip()
        
        if choice == "1":
            ars.run_continuous()
        elif choice == "2":
            ars.run_single_shot()
        elif choice == "3":
            ars.run_wake_word_mode()
        elif choice == "4":
            ars.test_wake_word_only()
        elif choice == "5":
            ars.test_model()
        else:
            print("Invalid choice. Using continuous mode.")
            ars.run_continuous()
            
    except Exception as e:
        logger.error(f"Failed to start Audio Response System: {e}")
        print(f"\nError: {e}")
        print("\nCommon solutions:")
        print("1. Ensure your microphone is working")
        print("2. Install required dependencies:")
        print("   conda install pytorch torchaudio -c pytorch")
        print("   pip install openwakeword pyaudio scipy numpy")
        print("3. Check microphone permissions on your system")

if __name__ == "__main__":
    main()