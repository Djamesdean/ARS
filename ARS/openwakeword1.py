# fixed_wake_word_detection.py
import numpy as np
import logging
from typing import Dict
import time
import collections

# Conditional import to handle missing OpenWakeWord
try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    print("Warning: OpenWakeWord not installed. Install with: pip install openwakeword")

import pyaudio

logger = logging.getLogger(__name__)

class FixedWakeWordDetector:
    def __init__(self, model_names=['alexa'], threshold=0.5, inference_framework='onnx'):
        """
        Initialize Wake Word Detector with proper audio handling.
        
        Args:
            model_names: List of wake word models to load
            threshold: Confidence threshold for wake word detection
            inference_framework: 'onnx' or 'tflite'
        """
        self.model_names = model_names
        self.threshold = threshold
        self.inference_framework = inference_framework
        self.model = None
        self.is_listening = False
        
        # Audio settings - These are CRITICAL for OpenWakeWord
        self.sample_rate = 16000  # Must be 16kHz
        self.chunk_size = 1280    # 80ms chunks (1280 samples at 16kHz)
        self.channels = 1
        
        # Prediction smoothing
        self.prediction_buffer_size = 5  # Keep last 5 predictions
        self.prediction_buffers = {name: collections.deque(maxlen=self.prediction_buffer_size) 
                                 for name in model_names}
        
        # Audio components
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the OpenWakeWord model with proper configuration."""
        if not OPENWAKEWORD_AVAILABLE:
            logger.error("OpenWakeWord is not available. Please install it first.")
            return False
        
        try:
            logger.info(f"Loading wake word models: {self.model_names}")
            logger.info(f"Using inference framework: {self.inference_framework}")
            
            # Load model with specific configuration
            self.model = Model(
                wakeword_models=self.model_names,
                inference_framework=self.inference_framework
            )
            
            logger.info("Wake word models loaded successfully")
            
            # Print model information
            for model_name in self.model_names:
                logger.info(f"Model '{model_name}' loaded and ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load wake word models: {e}")
            logger.info("Trying to download models...")
            
            try:
                # Force download if models aren't available
                self.model = Model(
                    wakeword_models=self.model_names,
                    inference_framework=self.inference_framework,
                    force_reload=True
                )
                logger.info("Models downloaded and loaded successfully")
                return True
            except Exception as e2:
                logger.error(f"Failed to download models: {e2}")
                return False
    
    def _normalize_audio(self, audio_data):
        """Properly normalize audio data for OpenWakeWord."""
        # Convert to float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # If audio is in int16 range, normalize to [-1, 1]
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0
        
        # Ensure values are in proper range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
    
    def detect_wake_word_single(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Detect wake word in a single audio chunk.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dict with model names as keys and confidence scores as values
        """
        if self.model is None:
            return {}
        
        try:
            # Normalize audio
            audio_data = self._normalize_audio(audio_data)
            
            # Ensure correct chunk size
            if len(audio_data) != self.chunk_size:
                if len(audio_data) < self.chunk_size:
                    # Pad with zeros
                    audio_data = np.pad(audio_data, (0, self.chunk_size - len(audio_data)))
                else:
                    # Take first chunk_size samples
                    audio_data = audio_data[:self.chunk_size]
            
            # Get predictions from model
            predictions = self.model.predict(audio_data)
            
            # Add to prediction buffers for smoothing
            for model_name, score in predictions.items():
                if model_name in self.prediction_buffers:
                    self.prediction_buffers[model_name].append(score)
            
            return predictions
                
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return {}
    
    def get_smoothed_predictions(self) -> Dict[str, float]:
        """Get smoothed predictions using recent history."""
        smoothed = {}
        for model_name, buffer in self.prediction_buffers.items():
            if len(buffer) > 0:
                # Use maximum of recent predictions
                smoothed[model_name] = max(buffer)
        return smoothed
    
    def start_listening_simple(self):
        """Start simple wake word detection using PyAudio directly."""
        if not OPENWAKEWORD_AVAILABLE or self.model is None:
            logger.error("Cannot start listening: Wake word model not available")
            return
        
        if self.is_listening:
            logger.warning("Already listening for wake words")
            return
        
        try:
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=None  # We'll use blocking read
            )
            
            self.is_listening = True
            logger.info(f"Started listening for wake words: {self.model_names}")
            logger.info(f"Chunk size: {self.chunk_size}, Sample rate: {self.sample_rate}")
            logger.info("Say the wake word loudly and clearly...")
            
            # Main listening loop
            consecutive_detections = 0
            while self.is_listening:
                try:
                    # Read audio data
                    audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Check if there's any audio activity
                    audio_level = np.mean(np.abs(audio_array))
                    
                    if audio_level > 100:  # Only process if there's some audio
                        # Get predictions
                        predictions = self.detect_wake_word_single(audio_array)
                        
                        # Check for wake word detection
                        detected = False
                        for model_name, score in predictions.items():
                            # Lower threshold for debugging
                            if score > 0.01:  # Very low threshold to see all activity
                                print(f"Audio activity - {model_name}: {score:.4f} (level: {audio_level:.0f})")
                            
                            if score > self.threshold:
                                logger.info(f"🎯 WAKE WORD DETECTED: '{model_name}' - Confidence: {score:.4f}")
                                detected = True
                                consecutive_detections += 1
                                
                                # If we get multiple consecutive detections, it's definitely the wake word
                                if consecutive_detections >= 2:
                                    logger.info(f"🚨 CONFIRMED WAKE WORD: '{model_name}' detected {consecutive_detections} times!")
                                    self._on_wake_word_detected(model_name, score)
                                    consecutive_detections = 0  # Reset counter
                        
                        if not detected:
                            consecutive_detections = 0
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error in audio processing: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Stop wake word detection and clean up resources."""
        self.is_listening = False
        
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
        
        logger.info("Stopped listening for wake words")
    
    def _on_wake_word_detected(self, model_name: str, confidence: float):
        """
        Callback function when wake word is detected.
        """
        print(f"\n{'='*50}")
        print("🎯 WAKE WORD ACTIVATED!")
        print(f"Model: {model_name}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"{'='*50}\n")
        
        # Here you can add your custom logic:
        # - Activate voice recognition
        # - Start recording for command detection
        # - Trigger other systems
        # - Play confirmation sound
    
    def test_audio_levels(self, duration=10):
        """Test audio input levels to help with debugging."""
        logger.info(f"Testing audio levels for {duration} seconds...")
        logger.info("Speak normally and watch the audio levels...")
        
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1280
            )
            
            start_time = time.time()
            max_level = 0
            
            while time.time() - start_time < duration:
                data = stream.read(1280, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                level = np.mean(np.abs(audio_array))
                max_level = max(max_level, level)
                
                if level > 100:
                    print(f"Audio level: {level:.0f} {'🔊' if level > 1000 else '🔉' if level > 500 else '🔈'}")
                
                time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            logger.info(f"Test completed. Maximum audio level: {max_level:.0f}")
            if max_level < 500:
                logger.warning("Audio levels seem low. Try speaking louder or check microphone settings.")
            
        except Exception as e:
            logger.error(f"Audio test failed: {e}")

def test_wake_word_detection():
    """Comprehensive test of wake word detection."""
    print("Wake Word Detection Test")
    print("=" * 40)
    
    # Test 1: Audio levels
    detector = FixedWakeWordDetector(model_names=['alexa'], threshold=0.3)  # Lower threshold
    
    if detector.model is None:
        print("❌ Failed to load wake word model")
        return
    
    print("\n1. Testing audio input levels...")
    print("Speak normally for the next 5 seconds:")
    detector.test_audio_levels(5)
    
    # Test 2: Wake word detection
    print("\n2. Starting wake word detection...")
    print("Say 'Alexa' clearly several times:")
    print("(The detector will show audio activity and wake word scores)")
    
    try:
        detector.start_listening_simple()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        detector.stop_listening()
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_wake_word_detection()