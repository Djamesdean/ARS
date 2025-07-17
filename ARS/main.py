# main.py
import time
import logging
import numpy as np
from config import *
from audio_capture import AudioCapture
from vad import SileroVAD
from noise_detection import noise_filter, calculate_energy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioResponseSystem:
    def __init__(self):
        """Initialize the Audio Response System with Silero VAD only."""
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
        
        print("\nAudio Response System - Silero VAD Only")
        print("=" * 40)
        print("1. Continuous processing (recommended)")
        print("2. Single-shot processing")
        print("3. Test model")
        print("=" * 40)
        
        choice = input("Choose mode (1-3): ").strip()
        
        if choice == "1":
            ars.run_continuous()
        elif choice == "2":
            ars.run_single_shot()
        elif choice == "3":
            ars.test_model()
        else:
            print("Invalid choice. Using continuous mode.")
            ars.run_continuous()
            
    except Exception as e:
        logger.error(f"Failed to start Audio Response System: {e}")
        print(f"\nError: {e}")
        print("\nCommon solutions:")
        print("1. Ensure your microphone is working")
        print("2. Check that the model file exists at: models/silero_vad.pth")
        print("3. Install required dependencies: pip install torch numpy scipy pyaudio")

if __name__ == "__main__":
    main()