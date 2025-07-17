
# config.py

# Model paths
SILERO_VAD_MODEL_PATH = "models/silero_vad.pth"

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 512 for Silero VAD compatibility (16000 Hz)
CHANNELS = 1

# VAD settings
VOICE_THRESHOLD = 0.5  # Silero VAD confidence threshold
ENERGY_THRESHOLD = 100  # Minimum energy to process audio