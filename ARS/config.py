# config.py

# Audio capture parameters
SAMPLE_RATE = 16000  # Sample rate in Hz
CHUNK_SIZE = 1024    # Size of each audio chunk captured

# VAD parameters
VAD_MODE = 3         # WebRTC VAD mode (0: aggressive, 3: least aggressive)

# Noise detection parameters
NOISE_THRESHOLD = 0.05  # Threshold for detecting high noise levels

# Model paths
SILERO_VAD_MODEL_PATH = "models.silero_vad.pth"  # Path to the SileroVAD model file

# Keyword detection parameters (future use)
KEYWORDS = ["Stop", "Emergency", "Cancel"]  # List of critical commands
