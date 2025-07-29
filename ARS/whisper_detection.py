# whisper_detection.py
import requests
import numpy as np
import wave
import os
import time
import logging
from typing import Optional

from audio_capture import AudioCapture
from noise_detection import noise_filter, calculate_energy
from config import SAMPLE_RATE, GROQ_API_KEY

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        """Initialize the Whisper transcriber with Groq API."""
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self.model_name = "whisper-large-v3"
        logger.info("WhisperTranscriber initialized with Groq API")
    
    def save_audio_to_wav(self, audio_data: np.ndarray, filename: str = "temp_audio.wav") -> str:

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)  # Use sample rate from config
                wf.writeframes(audio_data.tobytes())
            
            logger.debug(f"Audio saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio to WAV: {e}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:

        temp_filename = None
        
        try:
            # Save audio to temporary WAV file
            temp_filename = self.save_audio_to_wav(audio_data)
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
            }
            
            # Open file and send to API
            with open(temp_filename, "rb") as audio_file:
                files = {
                    "file": (temp_filename, audio_file, "audio/wav")
                }
                
                data = {
                    "model": self.model_name,
                    "response_format": "json"
                }
                
                logger.debug("Sending audio to Groq API...")
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    files=files, 
                    data=data,
                    timeout=30  # 30 second timeout
                )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                transcription = result.get("text", "").strip()
                logger.info(f"Transcription successful: '{transcription}'")
                return transcription if transcription else None
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.debug(f"Cleaned up {temp_filename}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_filename}: {e}")

def capture_audio_for_duration(duration_seconds: float = 3.0, apply_filtering: bool = True) -> np.ndarray:

    audio_capture = AudioCapture(
        sample_rate=SAMPLE_RATE,
        chunk_size=512,  # Your chunk size
        channels=1
    )
    
    # Calculate how many chunks we need
    # Each chunk is 512 samples, at 16kHz that's 512/16000 = 0.032 seconds per chunk
    chunk_duration = 512 / SAMPLE_RATE
    num_chunks = int(duration_seconds / chunk_duration)
    
    logger.info(f"Recording for {duration_seconds} seconds ({num_chunks} chunks)...")
    logger.info("Start speaking now!")
    
    audio_chunks = []
    
    try:
        for i in range(num_chunks):
            # Capture one chunk
            chunk = audio_capture.capture_audio()
            
            if len(chunk) > 0:
                # Apply noise filtering if requested
                if apply_filtering:
                    chunk = noise_filter(chunk, SAMPLE_RATE)
                
                audio_chunks.append(chunk)
                
                # Show progress every 10 chunks (about 0.32 seconds)
                if (i + 1) % 10 == 0:
                    elapsed = (i + 1) * chunk_duration
                    logger.debug(f"Recording progress: {elapsed:.1f}s / {duration_seconds}s")
            else:
                logger.warning(f"Empty chunk received at iteration {i}")
        
        if audio_chunks:
            # Concatenate all chunks
            full_audio = np.concatenate(audio_chunks)
            logger.info(f"Recording complete! Captured {len(full_audio)} samples ({len(full_audio)/SAMPLE_RATE:.2f}s)")
            
            # Check audio quality
            energy = calculate_energy(full_audio)
            logger.info(f"Audio energy level: {energy:.2f}")
            
            return full_audio
        else:
            logger.error("No audio data captured!")
            return np.array([], dtype=np.int16)
            
    except Exception as e:
        logger.error(f"Error during audio capture: {e}")
        return np.array([], dtype=np.int16)

def capture_audio_for_duration_continuous(duration_seconds: float = 3.0, apply_filtering: bool = True) -> np.ndarray:

    audio_capture = AudioCapture(
        sample_rate=SAMPLE_RATE,
        chunk_size=512,
        channels=1
    )
    
    logger.info(f"Recording for {duration_seconds} seconds using continuous method...")
    logger.info("Start speaking now!")
    
    audio_chunks = []
    
    try:
        # Start continuous capture
        audio_capture.start_continuous_capture()
        
        start_time = time.time()
        chunk_count = 0
        
        while time.time() - start_time < duration_seconds:
            audio_data = audio_capture.get_audio_data()
            if audio_data is not None:
                chunk_count += 1
                
                # Apply noise filtering if requested
                if apply_filtering:
                    audio_data = noise_filter(audio_data, SAMPLE_RATE)
                
                audio_chunks.append(audio_data)
                
                # Show progress every 50 chunks
                if chunk_count % 50 == 0:
                    elapsed = time.time() - start_time
                    energy = calculate_energy(audio_data)
                    logger.debug(f"Progress: {elapsed:.1f}s, chunks: {chunk_count}, energy: {energy:.1f}")
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
        
        # Stop capture
        audio_capture.stop_capture()
        
        if audio_chunks:
            # Concatenate all chunks
            full_audio = np.concatenate(audio_chunks)
            logger.info(f"Recording complete! Captured {len(full_audio)} samples ({len(full_audio)/SAMPLE_RATE:.2f}s)")
            
            # Check audio quality
            energy = calculate_energy(full_audio)
            logger.info(f"Audio energy level: {energy:.2f}")
            
            return full_audio
        else:
            logger.error("No audio data captured with continuous method!")
            return np.array([], dtype=np.int16)
            
    except Exception as e:
        logger.error(f"Error during continuous audio capture: {e}")
        return np.array([], dtype=np.int16)
    finally:
        # Make sure to stop capture
        try:
            audio_capture.stop_capture()
        except:
            pass


def capture_and_transcribe_continuous(duration_seconds: float = 3.0, apply_filtering: bool = True) -> Optional[str]:

    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Capture audio using continuous method
        audio_data = capture_audio_for_duration_continuous(duration_seconds, apply_filtering)
        
        if len(audio_data) == 0:
            logger.error("No audio data captured")
            return None
        
        # Check if audio has sufficient energy (not just silence)
        energy = calculate_energy(audio_data)
        if energy < 10:  # Lower threshold for testing
            logger.warning(f"Audio energy low ({energy:.2f}), but trying transcription anyway...")
        
        # Transcribe
        transcription = transcriber.transcribe_audio(audio_data)
        
        return transcription
        
    except Exception as e:
        logger.error(f"Error in capture_and_transcribe_continuous: {e}")
        return None

def test_whisper_api_with_debug():
    """Test Whisper API using both methods."""
    print("\n" + "="*50)
    print("WHISPER API TEST (CONTINUOUS METHOD)")
    print("="*50)
    
    try:
        
        print("Recording for 3 seconds - SPEAK NOW!")
        
        # Use continuous capture method
        transcription = capture_and_transcribe_continuous(duration_seconds=3.0)
        
        if transcription:
            print(f"✅ Success! Transcription: '{transcription}'")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")



if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_whisper_api_with_debug()