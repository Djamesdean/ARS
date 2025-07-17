
import numpy as np
import logging
from typing import Dict, Any, Optional
import time
from openwakeword import Model

logger = logging.getLogger(__name__)

class WakeWordDetector:
    def __init__(self, wake_word: str = "black", threshold: float = 0.5, 
                 sample_rate: int = 16000, proxy_model: str = "hey_jarvis"):
        """
        OpenWakeWord-based wake word detector.
        
        Since we don't have a specific "black" model, we'll use an existing model
        as a proxy and map its detection to our "black" wake word.
        
        Args:
            wake_word: Target wake word (what we want to detect)
            threshold: Detection threshold (0.0 to 1.0)
            sample_rate: Audio sample rate (must be 16000 for OpenWakeWord)
            proxy_model: Which OpenWakeWord model to use as proxy
        """
        self.wake_word = wake_word
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.proxy_model = proxy_model
        self.detection_callback = None
        self.is_initialized = False
        
        # Detection timing
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # 2 seconds between detections
        
        # Initialize OpenWakeWord
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize OpenWakeWord model."""
        try:
            logger.info("Initializing OpenWakeWord model...")
            
            # Create model with ONNX inference
            self.model = Model(inference_framework='onnx')
            print("Loaded models:", self.model.models.keys())
            # Check available models
            available_models = list(self.model.models.keys())
            logger.info(f"Available models: {available_models}")
            
            # Verify our chosen proxy model exists
            if self.proxy_model not in available_models:
                # Fallback to first available model
                self.proxy_model = available_models[0] if available_models else None
                logger.warning(f"Proxy model not found, using: {self.proxy_model}")
            
            if self.proxy_model:
                logger.info("✅ OpenWakeWord initialized successfully")
                logger.info(f"Using '{self.proxy_model}' model as proxy for '{self.wake_word}'")
                logger.info(f"When '{self.proxy_model}' is detected, we'll treat it as '{self.wake_word}'")
                self.is_initialized = True
            else:
                logger.error("No models available")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord: {e}")
            self.is_initialized = False
    
    def detect_wake_word(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect wake word using OpenWakeWord.
        
        Args:
            audio_data: Audio data as numpy array (int16 format)
            
        Returns:
            dict: Detection results
        """
        if not self.is_initialized:
            return self._create_result(False, 0.0, "not_initialized")
        
        try:
            # Check detection cooldown
            current_time = time.time()
            if current_time - self.last_detection_time < self.detection_cooldown:
                return self._create_result(False, 0.0, "cooldown")
            
            # Convert audio format for OpenWakeWord
            if audio_data.dtype == np.int16:
                # Convert int16 to float32 in range [-1, 1]
                audio_float = audio_data.astype(np.float32) / 32768.0
                audio_data = self._prepare_audio_for_oww(audio_float)
            else:
                audio_float = audio_data.astype(np.float32)
            
            # OpenWakeWord expects specific chunk sizes
            # Let's buffer audio to get the right size
            audio_processed = self._prepare_audio_for_oww(audio_float)
            
            if audio_processed is None:
                return self._create_result(False, 0.0, "audio_too_short")
            
            # Run OpenWakeWord prediction
            predictions = self.model.predict(audio_processed)
            
            # Get confidence for our proxy model
            proxy_confidence = predictions.get(self.proxy_model, 0.0)
            
            # Check if detection threshold is met
            detected = proxy_confidence > self.threshold
            
            # Log detection attempts
            if proxy_confidence > 0.1:  # Log significant attempts
                logger.debug(f"OpenWakeWord - {self.proxy_model}: {proxy_confidence:.3f}, "
                           f"threshold: {self.threshold}, detected: {detected}")
            
            if detected:
                self.last_detection_time = current_time
                logger.info(f"🎯 Wake word '{self.wake_word}' detected! "
                          f"(via {self.proxy_model} model: {proxy_confidence:.3f})")
                
                if self.detection_callback:
                    result = self._create_result(detected, proxy_confidence, "openwakeword")
                    self.detection_callback(result)
            
            return self._create_result(detected, proxy_confidence, "openwakeword")
            
        except Exception as e:
            logger.error(f"OpenWakeWord detection error: {e}")
            return self._create_result(False, 0.0, f"error: {e}")
    
    def _prepare_audio_for_oww(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepare audio data for OpenWakeWord.
        
        OpenWakeWord expects specific audio lengths and formats.
        """
        try:
            # OpenWakeWord typically works with longer audio chunks
            # If our chunk is too short, we might need to buffer
            min_length = int(0.5 * self.sample_rate)  # 0.5 seconds minimum
            
            if len(audio_data) < min_length:
                # For real-time processing, pad with zeros if too short
                padding = min_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
            # Ensure audio is the right length for the model
            # Most OpenWakeWord models expect specific durations
            target_length = int(1.0 * self.sample_rate)  # 1 second
            
            if len(audio_data) > target_length:
                # Take the last 1 second of audio
                audio_data = audio_data[-target_length:]
            elif len(audio_data) < target_length:
                # Pad to reach target length
                padding = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preparation error: {e}")
            return None
    
    def _create_result(self, detected: bool, confidence: float, method: str) -> Dict[str, Any]:
        """Create detection result."""
        return {
            'detected': detected,
            'confidence': float(confidence),
            'wake_word': self.wake_word,
            'proxy_model': self.proxy_model,
            'timestamp': time.time(),
            'method': method
        }
    
    def set_detection_callback(self, callback_function):
        """Set callback function for when wake word is detected."""
        self.detection_callback = callback_function
        logger.info("Wake word detection callback registered")
    
    def update_threshold(self, new_threshold: float):
        """Update detection threshold."""
        if 0.0 <= new_threshold <= 1.0:
            old_threshold = self.threshold
            self.threshold = new_threshold
            logger.info(f"Threshold updated: {old_threshold} -> {new_threshold}")
        else:
            logger.warning("Threshold must be between 0.0 and 1.0")
    
    def change_proxy_model(self, new_proxy_model: str):
        """Change which OpenWakeWord model to use as proxy."""
        if self.model and new_proxy_model in self.model.models:
            old_model = self.proxy_model
            self.proxy_model = new_proxy_model
            logger.info(f"Proxy model changed: {old_model} -> {new_proxy_model}")
        else:
            logger.warning(f"Model '{new_proxy_model}' not available")
    
    def get_available_models(self) -> list:
        """Get list of available OpenWakeWord models."""
        if self.model:
            return list(self.model.models.keys())
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            'wake_word': self.wake_word,
            'proxy_model': self.proxy_model,
            'threshold': self.threshold,
            'sample_rate': self.sample_rate,
            'is_initialized': self.is_initialized,
            'available_models': self.get_available_models(),
            'method': 'openwakeword'
        }
    
    def test_detection(self):
        """Test the detection system."""
        logger.info("=== OpenWakeWord Integration Test ===")
        status = self.get_status()
        
        for key, value in status.items():
            logger.info(f"{key}: {value}")
        
        logger.info("✅ Using OpenWakeWord with pre-trained models")
        logger.info(f"💡 Say '{self.proxy_model}' and it will be detected as '{self.wake_word}'")
        logger.info("💡 You can also try saying other available wake words")

def create_wake_word_detector(wake_word: str = "black", 
                             threshold: float = 0.5,
                             proxy_model: str = "hey_jarvis") -> WakeWordDetector:
    """
    Create OpenWakeWord-based detector.
    
    Args:
        wake_word: What you want to detect (e.g., "black")
        threshold: Detection sensitivity (0.5 is good default)
        proxy_model: Which OpenWakeWord model to use as proxy
    """
    detector = WakeWordDetector(
        wake_word=wake_word, 
        threshold=threshold,
        proxy_model=proxy_model
    )
    detector.test_detection()
    return detector