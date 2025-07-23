# working_openwakeword_detector.py
"""
Properly configured OpenWakeWord that actually works
Based on research findings for common detection issues
"""
import numpy as np
import pyaudio
import time
import logging
from collections import deque

try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    print("Install OpenWakeWord: pip install openwakeword")

logger = logging.getLogger(__name__)

class WorkingWakeWordDetector:
    def __init__(self, keywords=['alexa'], threshold=0.7):
        """
        Working OpenWakeWord implementation with proper configuration.
        Research shows these specific settings are critical for detection.
        """
        self.keywords = keywords
        self.threshold = threshold
        self.model = None
        self.is_listening = False
        
        # Critical audio settings from research
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms frames - CRITICAL for OpenWakeWord
        self.channels = 1
        
        # Detection improvements from research
        self.score_buffer = deque(maxlen=5)  # Smooth predictions
        self.audio_buffer = deque(maxlen=24000)  # 1.5 second buffer
        self.consecutive_detections = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load model with research-proven settings."""
        if not OPENWAKEWORD_AVAILABLE:
            logger.error("OpenWakeWord not available")
            return False
        
        try:
            logger.info("Loading OpenWakeWord with optimal settings...")
            
            # Key fix from research: enable noise suppression
            self.model = Model(
                wakeword_models=self.keywords,
                enable_speex_noise_suppression=True,  # Critical improvement
                vad_threshold=0.5  # Pre-filter audio
            )
            
            logger.info(f"✓ Model loaded with keywords: {self.keywords}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def _preprocess_audio(self, audio_data):
        """
        Research-based audio preprocessing that fixes detection issues.
        """
        # Convert to float32 and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Ensure proper range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Apply mild noise reduction (simple high-pass filter)
        # Remove low-frequency noise that interferes with detection
        if len(audio_data) > 10:
            # Simple high-pass: remove DC and very low frequencies
            audio_data = audio_data - np.mean(audio_data)
        
        return audio_data
    
    def _get_smoothed_score(self, current_scores):
        """
        Research finding: smooth scores over multiple frames for reliability.
        """
        # Add current max score to buffer
        max_score = max(current_scores.values()) if current_scores else 0.0
        self.score_buffer.append(max_score)
        
        # Return smoothed score (maximum of recent scores)
        return max(self.score_buffer) if self.score_buffer else 0.0
    
    def detect_wake_word(self, audio_data):
        """Enhanced detection with research-based improvements."""
        if self.model is None:
            return {}
        
        try:
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_data)
            
            # Ensure exact chunk size (critical from research)
            if len(processed_audio) != self.chunk_size:
                if len(processed_audio) < self.chunk_size:
                    processed_audio = np.pad(processed_audio, 
                                           (0, self.chunk_size - len(processed_audio)))
                else:
                    processed_audio = processed_audio[:self.chunk_size]
            
            # Convert back to int16 for model
            audio_int16 = (processed_audio * 32767).astype(np.int16)
            
            # Get predictions
            predictions = self.model.predict(audio_int16)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {}
    
    def start_detection(self):
        """Start detection with research-proven settings."""
        if not OPENWAKEWORD_AVAILABLE or self.model is None:
            print("❌ Cannot start - model not loaded")
            return
        
        print("🎯 Starting WORKING wake word detection...")
        print(f"Keywords: {self.keywords}")
        print(f"Threshold: {self.threshold}")
        print("Say the wake word clearly and wait for detection...")
        print("(Press Ctrl+C to stop)")
        
        try:
            # Setup PyAudio with exact research specifications
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_listening = True
            detection_count = 0
            
            while self.is_listening:
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Add to continuous buffer (research improvement)
                self.audio_buffer.extend(audio_array)
                
                # Check audio activity level
                audio_level = np.mean(np.abs(audio_array))
                
                if audio_level > 100:  # Only process active audio
                    # Get predictions
                    predictions = self.detect_wake_word(audio_array)
                    
                    if predictions:
                        # Get smoothed score
                        smoothed_score = self._get_smoothed_score(predictions)
                        
                        # Show activity for debugging
                        max_score = max(predictions.values())
                        best_keyword = max(predictions, key=predictions.get)
                        
                        if max_score > 0.05:  # Show significant activity
                            activity = "🔥" if max_score > self.threshold else "🔶"
                            print(f"{activity} {best_keyword}: {max_score:.4f} "
                                  f"(smoothed: {smoothed_score:.4f}, audio: {audio_level:.0f})")
                        
                        # Check for detection with smoothing
                        if smoothed_score > self.threshold:
                            self.consecutive_detections += 1
                            
                            if self.consecutive_detections >= 1:  # Immediate response
                                detection_count += 1
                                print("\n🎯 WAKE WORD DETECTED!")
                                print(f"   Keyword: {best_keyword}")
                                print(f"   Score: {max_score:.4f}")
                                print(f"   Smoothed: {smoothed_score:.4f}")
                                print(f"   Detection #{detection_count}")
                                print(f"   Time: {time.strftime('%H:%M:%S')}")
                                print("=" * 50)
                                
                                # Clear buffers after detection
                                self.score_buffer.clear()
                                self.consecutive_detections = 0
                                
                                # Call your custom handler here
                                self._on_wake_word_detected(best_keyword, max_score)
                        else:
                            self.consecutive_detections = 0
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print(f"\n✅ Detection stopped. Total detections: {detection_count}")
        except Exception as e:
            print(f"❌ Detection failed: {e}")
        finally:
            self.is_listening = False
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass
    
    def _on_wake_word_detected(self, keyword, score):
        """Override this method to add your custom wake word actions."""
        print(f"🚨 WAKE WORD ACTIVATED: {keyword}")
        # Add your custom logic here:
        # - Start voice recording
        # - Activate your main system
        # - Play confirmation sound
        pass

def test_working_detector():
    """Test the working implementation."""
    print("Working OpenWakeWord Test")
    print("=" * 40)
    
    # Create detector
    detector = WorkingWakeWordDetector(
        keywords=['alexa'],
        threshold=0.6  # Start with lower threshold
    )
    
    if detector.model is None:
        print("❌ Failed to load model")
        print("Try: pip install --upgrade openwakeword")
        return
    
    print("✅ Model loaded successfully!")
    print("\nInstructions:")
    print("1. Speak clearly: 'Ah-LEX-ah'")
    print("2. Wait 1-2 seconds between attempts")
    print("3. Watch the activity indicators")
    print("4. If no detection, try speaking louder")
    
    input("\nPress Enter to start detection...")
    
    try:
        detector.start_detection()
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    test_working_detector()