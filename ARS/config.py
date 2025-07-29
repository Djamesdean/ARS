
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

OPENAI_API_KEY = "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"  