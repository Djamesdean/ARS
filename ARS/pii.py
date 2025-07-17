# mic_test.py
import pyaudio
import numpy as np

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)

print("Recording one chunk...")
data = stream.read(512)
audio = np.frombuffer(data, dtype=np.int16)
print(f"Mean volume: {np.mean(np.abs(audio))}")

stream.close()
p.terminate()
