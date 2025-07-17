# vad.py
import torch
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

class SileroVAD:
    def __init__(self, use_local_cache: bool = True):
        self.model = None
        self.sample_rate = 16000
        
        try:
            logger.info("Loading Silero VAD model...")
            
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            
            self.model.eval()
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise
    
    def detect_voice(self, audio_data: Union[np.ndarray, torch.Tensor]) -> float:
        if self.model is None:
            raise RuntimeError("Silero VAD model not loaded")
        
        try:
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            else:
                audio_tensor = audio_data.float()
            
            if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
                audio_tensor = audio_tensor / 32768.0
            
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            expected_size = 512
            current_size = audio_tensor.shape[1]
            
            if current_size != expected_size:
                if current_size < expected_size:
                    padding = expected_size - current_size
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                else:
                    audio_tensor = audio_tensor[:, :expected_size]
            
            with torch.no_grad():
                voice_prob = self.model(audio_tensor, self.sample_rate).squeeze().item()
            
            return voice_prob
            
        except Exception as e:
            logger.error(f"Error in Silero VAD: {e}")
            return 0.0