# test_oww_integration.py - Test OpenWakeWord Integration
import time
import logging
import sys
import os
import numpy as np
# Add current directory to path to import our modules
sys.path.append(os.getcwd())

from wake_word import create_wake_word_detector
from audio_capture import AudioCapture
from config import SAMPLE_RATE, CHUNK_SIZE, CHANNELS

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def test_openwakeword_integration():
    """Test OpenWakeWord integration with your audio system."""
    
    print("\n" + "=" * 60)
    print("🎯 OPENWAKEWORD INTEGRATION TEST")
    print("=" * 60)
    
    # Test different proxy models
    available_proxies = ["hey_jarvis", "alexa", "hey_mycroft"]
    
    print("Available proxy models to test:")
    for i, model in enumerate(available_proxies, 1):
        print(f"{i}. {model}")
    
    choice = input("\nChoose proxy model (1-3) or press Enter for hey_jarvis: ").strip()
    
    if choice == "2":
        proxy_model = "alexa"
    elif choice == "3":
        proxy_model = "hey_mycroft"
    else:
        proxy_model = "hey_jarvis"
    
    print(f"\nUsing '{proxy_model}' as proxy for 'black' detection")
    
    # Create detector
    try:
        detector = create_wake_word_detector(
            wake_word="black",
            threshold=0.5,
            proxy_model=proxy_model
        )
        print("✅ OpenWakeWord detector created successfully")
    except Exception as e:
        print(f"❌ Failed to create detector: {e}")
        return
    
    # Create audio capture
    try:
        audio_capture = AudioCapture(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            channels=CHANNELS
        )
        print("✅ Audio capture created successfully")
    except Exception as e:
        print(f"❌ Failed to create audio capture: {e}")
        return
    
    # Set up detection callback
    detection_count = 0
    
    def on_detection(result):
        nonlocal detection_count
        detection_count += 1
        print(f"\n🎉 DETECTION #{detection_count}")
        print(f"Wake word: {result['wake_word']}")
        print(f"Proxy model: {result['proxy_model']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print("-" * 40)
    
    detector.set_detection_callback(on_detection)
    
    # Show instructions
    print("\n📋 INSTRUCTIONS:")
    print(f"1. Say '{proxy_model}' to trigger detection")
    print("2. The system will detect it as 'black' wake word")
    print("3. Try different volumes and distances")
    print("4. Press Ctrl+C to stop")
    print(f"\n🎤 Listening for '{proxy_model}'...")
    print("=" * 60)
    
    # Audio processing loop
    try:
        chunk_count = 0
        while True:
            # Buffer 1 second of audio (16,000 samples)
            buffered_audio = []

            while len(buffered_audio) < SAMPLE_RATE:
                chunk = audio_capture.capture_audio()
                if len(chunk) > 0:
                      buffered_audio.extend(chunk.tolist())

            audio_np = np.array(buffered_audio[-SAMPLE_RATE:], dtype=np.int16)
            result = detector.detect_wake_word(audio_np)
                
                # Show periodic status
            if chunk_count % 50 == 0:  # Every ~1.6 seconds
                    confidence = result.get('confidence', 0)
                    if confidence > 0.1:
                        print(f"Listening... (recent confidence: {confidence:.1%})")
                    else:
                        print("Listening... (no significant audio)")
            
            time.sleep(0.05)  # Small delay
            
    except KeyboardInterrupt:
        print("\n\n📊 TEST RESULTS:")
        print(f"Total detections: {detection_count}")
        print(f"Proxy model used: {proxy_model}")
        print("Target wake word: black")
        print("Test completed!")
    
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

def test_all_models():
    """Test all available models quickly."""
    print("\n" + "=" * 60)
    print("🔍 TESTING ALL AVAILABLE MODELS")
    print("=" * 60)
    
    # Create detector to get available models
    detector = create_wake_word_detector()
    available_models = detector.get_available_models()
    
    print(f"Available models: {available_models}")
    
    for model in available_models[:3]:  # Test first 3 models
        print(f"\nTesting {model}...")
        try:
            test_detector = create_wake_word_detector(proxy_model=model)
            print(f"✅ {model} initialized successfully")
        except Exception as e:
            print(f"❌ {model} failed: {e}")

if __name__ == "__main__":
    print("OpenWakeWord Integration Test")
    print("1. Test specific model")
    print("2. Test all models")
    
    choice = input("Choose option (1-2): ").strip()
    
    if choice == "2":
        test_all_models()
    else:
        test_openwakeword_integration()