# debug_wakeword.py
"""
Step-by-step debugging script for wake word detection
"""

import numpy as np
import pyaudio
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_microphone():
    """Test basic microphone functionality."""
    print("\n" + "="*50)
    print("STEP 1: Testing Microphone")
    print("="*50)
    
    try:
        p = pyaudio.PyAudio()
        
        print("Available audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']} ({info['maxInputChannels']} channels)")
        
        # Test recording
        print("\nTesting microphone recording...")
        print("Speak loudly for 3 seconds:")
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280
        )
        
        max_level = 0
        for i in range(30):  # 3 seconds at ~10 FPS
            data = stream.read(1280, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            level = np.mean(np.abs(audio_array))
            max_level = max(max_level, level)
            
            # Visual feedback
            bars = "█" * min(20, int(level / 100))
            print(f"\rAudio: {level:6.0f} |{bars:<20}|", end="", flush=True)
            time.sleep(0.1)
        
        print("\n\nMicrophone test results:")
        print(f"  Maximum audio level: {max_level:.0f}")
        
        if max_level < 100:
            print("  ⚠️  WARNING: Audio levels very low!")
            print("     - Check microphone connection")
            print("     - Increase microphone volume")
            print("     - Grant microphone permissions")
        elif max_level < 500:
            print("  ⚠️  Audio levels low. Try speaking louder.")
        else:
            print("  ✅ Microphone working well!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return max_level > 100
        
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False

def test_openwakeword_import():
    """Test OpenWakeWord import and basic functionality."""
    print("\n" + "="*50)
    print("STEP 2: Testing OpenWakeWord")
    print("="*50)
    
    try:
        from openwakeword.model import Model
        print("✅ OpenWakeWord imported successfully")
        
        # Test model loading
        print("Loading 'alexa' model...")
        model = Model(wakeword_models=['alexa'])
        print("✅ Model loaded successfully")
        
        # Test with dummy audio
        dummy_audio = np.random.randn(1280).astype(np.float32) * 0.001  # Very quiet noise
        predictions = model.predict(dummy_audio)
        print(f"✅ Model prediction test: {predictions}")
        
        return model
        
    except ImportError:
        print("❌ OpenWakeWord not installed")
        print("Install with: pip install openwakeword")
        return None
    except Exception as e:
        print(f"❌ OpenWakeWord test failed: {e}")
        return None

def test_wake_word_with_real_audio(model):
    """Test wake word detection with real audio input."""
    print("\n" + "="*50)
    print("STEP 3: Testing Wake Word Detection")
    print("="*50)
    
    if model is None:
        print("❌ Cannot test - model not loaded")
        return
    
    try:
        print("Starting wake word detection test...")
        print("Instructions:")
        print("1. Say 'Alexa' clearly and loudly")
        print("2. Wait 1-2 seconds between attempts")
        print("3. Try different pronunciations: 'Ah-LEX-ah'")
        print("4. Press Ctrl+C to stop")
        print("\nListening... (showing all scores > 0.01)")
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280
        )
        
        detection_count = 0
        total_frames = 0
        
        while True:
            # Read audio
            data = stream.read(1280, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Check audio level
            audio_level = np.mean(np.abs(audio_array))
            
            # Only process if there's audio activity
            if audio_level > 50:
                # Normalize audio properly
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Get predictions
                predictions = model.predict(audio_float)
                total_frames += 1
                
                # Show any significant activity
                for model_name, score in predictions.items():
                    if score > 0.01:  # Show low-level activity
                        activity_level = "🔥" if score > 0.5 else "🔶" if score > 0.1 else "🔸"
                        print(f"{activity_level} {model_name}: {score:.4f} (audio: {audio_level:.0f})")
                        
                        if score > 0.5:  # Actual detection
                            detection_count += 1
                            print("\n🎯 WAKE WORD DETECTED!")
                            print(f"   Model: {model_name}")
                            print(f"   Score: {score:.4f}")
                            print(f"   Detection #{detection_count}")
                            print(f"   Time: {time.strftime('%H:%M:%S')}")
                            print("=" * 40)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\nTest Summary:")
        print(f"  Total audio frames processed: {total_frames}")
        print(f"  Wake words detected: {detection_count}")
        print(f"  Detection rate: {detection_count/max(1,total_frames)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Wake word test failed: {e}")
    
    finally:
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass

def main():
    """Run complete diagnostic test."""
    print("OpenWakeWord Diagnostic Tool")
    print("="*50)
    print("This will test your microphone and wake word detection step by step.")
    
    # Step 1: Test microphone
    mic_ok = test_microphone()
    if not mic_ok:
        print("\n❌ STOPPED: Fix microphone issues first")
        return
    
    # Step 2: Test OpenWakeWord
    model = test_openwakeword_import()
    if model is None:
        print("\n❌ STOPPED: Fix OpenWakeWord installation first")
        return
    
    # Step 3: Test actual wake word detection
    input("\nPress Enter to start wake word detection test...")
    test_wake_word_with_real_audio(model)
    
    print("\n✅ Diagnostic complete!")
    print("\nTroubleshooting tips:")
    print("- If no audio detected: Check microphone permissions")
    print("- If audio detected but no wake word: Try speaking louder/clearer")
    print("- If wake word scores low: Try different pronunciations")
    print("- If still no detection: Lower threshold in main code")

if __name__ == "__main__":
    main()