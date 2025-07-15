# main.py

from audio_capture import AudioCapture
from vad import VAD, SileroVAD
from noise_detection import noise_filter, noise_level

def main():
    # Initialize capture and VAD models
    audio_capture = AudioCapture(sample_rate=16000)
    vad = VAD(vad_mode=3)  # Aggressive mode for WebRTC VAD
    silero_vad = SileroVAD(model_path="models.silero_vad.pth")

    print("Starting audio capture...")

    while True:
        # Capture audio
        audio_data = audio_capture.capture_audio()

        # Apply noise filtering
        filtered_audio = noise_filter(audio_data)

        # Detect noise level
        noise = noise_level(filtered_audio)

        if noise > 0.05:  # threshold for noise detection
            print("High noise detected, switching to Silero VAD for accuracy.")
            voice_detected = silero_vad.detect_voice(filtered_audio)
        else:
            voice_detected = vad.detect_voice(filtered_audio)
        
        if any(voice_detected):
            print("Voice detected, triggering critical command detection...")
            # You can add additional logic for keyword spotting and action here.

if __name__ == "__main__":
    main()
