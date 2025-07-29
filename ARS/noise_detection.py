
import numpy as np
import scipy.signal

def noise_filter(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:

    try:
        # Apply bandpass filter to focus on speech frequencies (300-3000 Hz)
        nyquist = 0.5 * sample_rate
        low = 300 / nyquist
        high = 3000 / nyquist
        
        # Ensure filter frequencies are valid
        low = max(low, 0.001)
        high = min(high, 0.999)
        
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
        
        return filtered_audio.astype(audio_data.dtype)
    except Exception as e:
        logging.error(f"Error in noise filtering: {e}")
        return audio_data

def calculate_energy(audio_data: np.ndarray) -> float:

    return np.mean(np.abs(audio_data.astype(np.float64)))

def spectral_subtraction(audio_data: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:

    if noise_profile is None or len(noise_profile) == 0:
        return audio_data
        
    try:
        # Simple spectral subtraction implementation
        fft_signal = np.fft.rfft(audio_data)
        fft_noise = np.fft.rfft(noise_profile)
        
        # Subtract noise spectrum
        magnitude = np.abs(fft_signal)
        noise_magnitude = np.abs(fft_noise)
        
        # Spectral subtraction with over-subtraction factor
        alpha = 2.0
        clean_magnitude = magnitude - alpha * noise_magnitude
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct signal
        phase = np.angle(fft_signal)
        clean_fft = clean_magnitude * np.exp(1j * phase)
        
        return np.fft.irfft(clean_fft, len(audio_data))
    except Exception as e:
        logging.error(f"Error in spectral subtraction: {e}")
        return audio_data