
import pyaudio
import numpy as np
import threading
import queue
import logging
from typing import Optional, Generator

logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512, 
                 channels: int = 1, format: int = pyaudio.paInt16):
 
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()
        self.p.terminate()

    def capture_audio(self) -> np.ndarray:

        try:
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            data = stream.read(self.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            stream.stop_stream()
            stream.close()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            return np.array([], dtype=np.int16)
    
    def start_continuous_capture(self) -> None:
        """Start continuous audio capture in a separate thread."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def stop_capture(self) -> None:
        """Stop continuous audio capture."""
        self.is_recording = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
    
    def _capture_loop(self) -> None:
        """Internal method for continuous capture loop."""
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                data = self.stream.read(self.chunk_size)
                audio_data = np.frombuffer(data, dtype=np.int16)
                self.audio_queue.put(audio_data)
                
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
            self.is_recording = False
    
    def get_audio_data(self) -> Optional[np.ndarray]:
  
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_audio_stream(self) -> Generator[np.ndarray, None, None]:
        """
        Generator for continuous audio stream.
        Yields:
            np.ndarray: Audio chunks
        """
        while self.is_recording:
            audio_data = self.get_audio_data()
            if audio_data is not None:
                yield audio_data
