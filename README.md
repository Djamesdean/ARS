# ARS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Audio-Based Recognition System for

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ARS and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ARS   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ARS a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Approach](#architecture--approach)
3. [File Structure](#file-structure)
4. [Detailed Code Analysis](#detailed-code-analysis)
5. [Development Journey & Issues](#development-journey--issues)
6. [Troubleshooting](#troubleshooting)
7. [Wake Word Detection Approaches](#wake-word-detection-approaches)
8. [Command Detection Approaches
](#command-detection-approaches)
9. [My Approach](#my-approach)


---

## Project Overview

The Audio Response System (ARS) is a real-time voice activity detection system designed to distinguish human speech from background noise and other audio sources. The system uses the Silero VAD (Voice Activity Detection) model to accurately detect when someone is speaking, making it suitable for voice assistants, smart home systems, and other voice-activated applications.

### Key Features
- **Real-time voice detection** using advanced deep learning models
- **Noise filtering** to improve detection accuracy
- **Energy-based preprocessing** to optimize performance
- **Two processing modes**: Continuous and single-shot
- **Automatic environment calibration** for optimal performance
- **Comprehensive logging** for debugging and monitoring

### Technical Specifications
- **Sample Rate**: 16,000 Hz (16 kHz)
- **Chunk Size**: 512 samples (32ms at 16kHz)
- **Audio Format**: 16-bit signed integer (int16)
- **Channels**: Mono (1 channel)
- **Model**: Silero VAD v3.1 (PyTorch JIT)

---

## Architecture & Approach

### Initial Dual-VAD Approach (Abandoned)
Initially, the system was designed with two VAD models:
- **WebRTC VAD**: Fast, lightweight, general audio detection
- **Silero VAD**: Accurate, speech-specific detection using deep learning

The idea was to switch between models based on noise levels, but this approach had several issues:
- Inconsistent behavior between models
- Complex threshold management
- WebRTC VAD detected any audio (not just speech)
- Calibration dependency

### Final Single-VAD Approach (Current)
The final implementation uses only **Silero VAD** with the following rationale:
- **Consistency**: Single model provides uniform behavior
- **Accuracy**: Silero VAD is specifically trained for human speech detection
- **Simplicity**: Easier to maintain and debug
- **Performance**: Energy pre-filtering provides sufficient speed optimization

### Processing Pipeline
```
Audio Input → Energy Check → Noise Filtering → Silero VAD → Voice Detection
```

1. **Audio Capture**: Continuous microphone input in 512-sample chunks
2. **Energy Pre-filtering**: Skip processing silent audio (< 100 energy threshold)
3. **Noise Filtering**: Bandpass filter (300-3000 Hz) for speech frequencies
4. **Voice Detection**: Silero VAD model inference
5. **Decision**: Binary voice/no-voice based on 0.5 confidence threshold

---

## File Structure

```
ARS/
├── config.py              # Configuration settings
├── vad.py                 # Voice Activity Detection implementation
├── noise_detection.py     # Audio filtering and processing
├── audio_capture.py       # Audio input handling
├── main.py               # Main application and system integration
└── README.md             # This documentation
```

**Note**: Model files are automatically cached by PyTorch Hub in `~/.cache/torch/hub/`

---

## Detailed Code Analysis

### 1. config.py
**Purpose**: Centralized configuration management for all system parameters.

```python
# Audio settings
SAMPLE_RATE = 16000        # Required by Silero VAD
CHUNK_SIZE = 512           # Optimized for 16kHz Silero VAD
CHANNELS = 1               # Mono audio input

# VAD settings  
VOICE_THRESHOLD = 0.5      # Silero confidence threshold
ENERGY_THRESHOLD = 100     # Skip silent audio below this energy
```

**Key Design Decisions**:
- **CHUNK_SIZE = 512**: Silero VAD specifically requires 512 samples for 16kHz audio
- **SAMPLE_RATE = 16000**: Standard rate for speech processing, required by Silero VAD
- **VOICE_THRESHOLD = 0.5**: Balanced threshold to avoid false positives/negatives
- **ENERGY_THRESHOLD = 100**: Empirically determined to skip silence while preserving quiet speech

### 2. vad.py
**Purpose**: Core voice activity detection using Silero VAD model.

#### Class: SileroVAD

**Initialization Process**:
```python
def __init__(self, use_local_cache: bool = True):
    self.model = None
    self.sample_rate = 16000
    
    # Load model using torch.hub caching
    self.model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,  # Use cache if available
        onnx=False,          # Use PyTorch JIT version
        verbose=False        # Reduce logging noise
    )
```

**Model Loading Strategy**:
- Uses `torch.hub.load()` with automatic caching
- First run downloads model to `~/.cache/torch/hub/`
- Subsequent runs load from cache instantly
- No internet required after initial download

**Voice Detection Process**:
```python
def detect_voice(self, audio_data: Union[np.ndarray, torch.Tensor]) -> float:
    # 1. Convert numpy array to PyTorch tensor
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
    
    # 2. Normalize int16 audio to [-1, 1] range
    if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
        audio_tensor = audio_tensor / 32768.0
    
    # 3. Ensure correct tensor shape (batch_size, samples)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # 4. Handle chunk size requirements
    expected_size = 512
    current_size = audio_tensor.shape[1]
    
    if current_size != expected_size:
        if current_size < expected_size:
            # Pad with zeros if too short
            padding = expected_size - current_size
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        else:
            # Truncate if too long
            audio_tensor = audio_tensor[:, :expected_size]
    
    # 5. Run inference
    with torch.no_grad():
        voice_prob = self.model(audio_tensor, self.sample_rate).squeeze().item()
    
    return voice_prob
```

**Critical Implementation Details**:
- **Normalization**: Converts int16 range (-32768 to 32767) to float range (-1.0 to 1.0)
- **Chunk Size Handling**: Automatically pads or truncates to exactly 512 samples
- **Batch Dimension**: Adds batch dimension for model compatibility
- **No Gradient**: Uses `torch.no_grad()` for inference optimization

### 3. noise_detection.py
**Purpose**: Audio preprocessing and noise reduction functions.

#### Function: noise_filter()
```python
def noise_filter(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    # Calculate Nyquist frequency
    nyquist = 0.5 * sample_rate
    
    # Define speech frequency band (300-3000 Hz)
    low = 300 / nyquist    # High-pass: Remove low-frequency noise
    high = 3000 / nyquist  # Low-pass: Remove high-frequency noise
    
    # Create 4th-order Butterworth bandpass filter
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    
    # Apply zero-phase filtering (forward and backward pass)
    filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
    
    return filtered_audio.astype(audio_data.dtype)
```

**Filter Design Rationale**:
- **300-3000 Hz**: Covers fundamental speech frequencies
- **4th-order Butterworth**: Good balance of filtering and minimal distortion
- **Zero-phase filtering**: Preserves timing relationships in audio
- **Removes**: Low-frequency rumble, high-frequency hiss/noise

#### Function: calculate_energy()
```python
def calculate_energy(audio_data: np.ndarray) -> float:
    return np.mean(np.abs(audio_data.astype(np.float64)))
```

**Energy Calculation**:
- Uses Mean Absolute Deviation (MAD) for computational efficiency
- Converts to float64 to prevent overflow
- Provides quick silence detection metric

#### Function: spectral_subtraction()
**Purpose**: Advanced noise reduction using spectral subtraction algorithm.

```python
def spectral_subtraction(audio_data: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
    # Transform to frequency domain
    fft_signal = np.fft.rfft(audio_data)
    fft_noise = np.fft.rfft(noise_profile)
    
    # Extract magnitude and phase
    magnitude = np.abs(fft_signal)
    noise_magnitude = np.abs(fft_noise)
    phase = np.angle(fft_signal)
    
    # Spectral subtraction with over-subtraction factor
    alpha = 2.0  # Over-subtraction factor
    clean_magnitude = magnitude - alpha * noise_magnitude
    
    # Prevent over-subtraction artifacts
    clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
    
    # Reconstruct signal
    clean_fft = clean_magnitude * np.exp(1j * phase)
    return np.fft.irfft(clean_fft, len(audio_data))
```

### 4. audio_capture.py
**Purpose**: Real-time audio input handling with threading support.

#### Class: AudioCapture

**Initialization**:
```python
def __init__(self, sample_rate: int = 16000, chunk_size: int = 512, 
             channels: int = 1, format: int = pyaudio.paInt16):
    self.sample_rate = sample_rate
    self.chunk_size = chunk_size
    self.channels = channels
    self.format = format
    self.p = pyaudio.PyAudio()
    self.audio_queue = queue.Queue()
    self.is_recording = False
```

**Single-Shot Capture**:
```python
def capture_audio(self) -> np.ndarray:
    # Open audio stream
    stream = self.p.open(
        format=self.format,
        channels=self.channels,
        rate=self.sample_rate,
        input=True,
        frames_per_buffer=self.chunk_size
    )
    
    # Read one chunk
    data = stream.read(self.chunk_size)
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    # Clean up
    stream.stop_stream()
    stream.close()
    
    return audio_data
```

**Continuous Capture with Threading**:
```python
def start_continuous_capture(self) -> None:
    if self.is_recording:
        return
        
    self.is_recording = True
    self.capture_thread = threading.Thread(target=self._capture_loop)
    self.capture_thread.daemon = True  # Dies with main thread
    self.capture_thread.start()

def _capture_loop(self) -> None:
    # Background thread continuously captures audio
    self.stream = self.p.open(...)
    
    while self.is_recording:
        data = self.stream.read(self.chunk_size)
        audio_data = np.frombuffer(data, dtype=np.int16)
        self.audio_queue.put(audio_data)  # Thread-safe queue
```

**Thread Safety**:
- Uses `queue.Queue()` for thread-safe data transfer
- Daemon threads automatically terminate with main process
- Proper resource cleanup in `stop_capture()`

### 5. main.py
**Purpose**: System integration, orchestration, and user interface.

#### Class: AudioResponseSystem

**Initialization and Dependencies**:
```python
def __init__(self):
    # Initialize audio capture
    self.audio_capture = AudioCapture(
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE,
        channels=CHANNELS
    )
    
    # Initialize Silero VAD with error handling
    try:
        self.silero_vad = SileroVAD()
        logger.info("Silero VAD initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Silero VAD: {e}")
        raise
```

**Environment Calibration**:
```python
def calibrate_noise(self, samples: int = 30) -> None:
    logger.info("Please stay quiet during calibration...")
    
    noise_data = []
    for i in range(samples):
        audio_data = self.audio_capture.capture_audio()
        if len(audio_data) > 0:
            noise_data.append(audio_data)
        time.sleep(0.1)  # 100ms between samples
    
    if noise_data:
        # Calculate noise profile for future use
        self.noise_profile = np.mean(noise_data, axis=0)
        avg_energy = np.mean([calculate_energy(data) for data in noise_data])
        logger.info(f"Average quiet energy: {avg_energy:.2f}")
```

**Core Processing Logic**:
```python
def process_audio_chunk(self, audio_data: np.ndarray) -> dict:
    # 1. Quick energy check (optimization)
    energy = calculate_energy(audio_data)
    if energy < ENERGY_THRESHOLD:
        return {
            'voice_detected': False,
            'voice_probability': 0.0,
            'energy_level': energy,
            'timestamp': time.time()
        }
    
    # 2. Apply noise filtering
    filtered_audio = noise_filter(audio_data, SAMPLE_RATE)
    
    # 3. Run VAD inference
    voice_prob = self.silero_vad.detect_voice(filtered_audio)
    voice_detected = voice_prob > VOICE_THRESHOLD
    
    # 4. Return comprehensive results
    return {
        'voice_detected': voice_detected,
        'voice_probability': voice_prob,
        'energy_level': energy,
        'timestamp': time.time()
    }
```

**Processing Modes**:

**Continuous Mode**:
- Uses background threading for real-time processing
- Processes audio continuously without blocking
- Suitable for always-on applications

**Single-Shot Mode**:
- Processes one chunk at a time
- Simpler execution model
- Good for testing and debugging

---

## Development Journey & Issues

### Phase 1: Initial Implementation Problems

#### Issue 1: Dual-VAD Complexity
**Problem**: Original implementation used both WebRTC VAD and Silero VAD with dynamic switching.

**Symptoms**:
- Inconsistent detection behavior
- Complex threshold management
- WebRTC VAD detecting non-speech audio (keyboard clicks, music)
- Environment-dependent calibration requirements

**Analysis**:
- WebRTC VAD designed for general audio activity, not speech-specific
- Threshold tuning became environment-specific
- Switching logic added unnecessary complexity

**Solution**: Simplified to Silero VAD only
```python
# Before: Complex dual-VAD logic
if noise > threshold and silero_available:
    use_silero_vad()
else:
    use_webrtc_vad()

# After: Simple single-VAD approach
voice_prob = silero_vad.detect_voice(audio_data)
voice_detected = voice_prob > 0.5
```

#### Issue 2: Chunk Size Incompatibility
**Problem**: Silero VAD requires specific chunk sizes but system used 1024 samples.

**Error Message**:
```
ERROR:vad:Error in Silero VAD: Provided number of samples is 1024 
(Supported values: 256 for 8000 sample rate, 512 for 16000)
```

**Root Cause**: Silero VAD v3.1 has strict input requirements:
- 8000 Hz → 256 samples
- 16000 Hz → 512 samples

**Solution**: Updated configuration and added automatic chunk size handling
```python
# config.py
CHUNK_SIZE = 512  # Changed from 1024

# vad.py - Automatic size adjustment
expected_size = 512
if current_size != expected_size:
    if current_size < expected_size:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
    else:
        audio_tensor = audio_tensor[:, :expected_size]
```

### Phase 2: Model Loading and Serialization Issues

#### Issue 3: Model Serialization Problems
**Problem**: Attempting to save Silero VAD model locally for offline use.

**Error Message**:
```
RuntimeError: Tried to serialize object torch.vad.model.vad_annotator.VADRNNJITMerge 
which does not have a getstate method defined!
```

**Root Cause**: Silero VAD uses PyTorch JIT (Just-In-Time) compilation, which creates models that cannot be serialized with standard `torch.save()`.

**Failed Approaches**:
1. Direct model saving: `torch.save(model, 'model.pth')`
2. State dict extraction and reloading
3. Manual model architecture recreation

**Solution**: Adopted torch.hub caching mechanism
```python
# Uses PyTorch's built-in caching system
self.model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,  # Use cache if available
    onnx=False
)
```

**Benefits of torch.hub caching**:
- Automatic model caching in `~/.cache/torch/hub/`
- No internet required after first download
- Handles JIT model serialization internally
- Cross-project model sharing

#### Issue 4: Collections.OrderedDict Error
**Problem**: Attempted to call `.eval()` on model state dictionary.

**Error Message**:
```
Error: 'collections.OrderedDict' object has no attribute 'eval'
```

**Root Cause**: Loaded `.pth` file contained model weights (OrderedDict) rather than complete model object.

**Solution**: Implemented proper model loading with architecture + weights
```python
# Detect if loaded object is state dict or complete model
if hasattr(self.model, 'eval'):
    self.model.eval()  # Complete model
else:
    # Load architecture and apply weights
    base_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    base_model.load_state_dict(loaded_state_dict)
    self.model = base_model
```
--- 

### Key Architectural Decisions and Rationale

#### Decision 1: Single VAD Model
**Rationale**: Simplicity and consistency over marginal performance gains
- Eliminated complex switching logic
- Provided uniform behavior across environments
- Reduced debugging complexity

#### Decision 2: Energy Pre-filtering
**Rationale**: Optimize performance without sacrificing accuracy
- Skip VAD processing for silent audio
- Significant CPU savings in quiet environments
- No impact on voice detection accuracy

#### Decision 3: torch.hub Caching
**Rationale**: Balance between offline capability and maintenance overhead
- Automatic model management
- No manual file handling
- Cross-project compatibility

#### Decision 4: 512-sample Chunks
**Rationale**: Model requirements override performance preferences
- Silero VAD strict requirement
- 32ms latency acceptable for real-time applications
- Optimal model performance

---

### Processing Modes Explained

#### Mode 1: Continuous Processing (Recommended)

**Best for**: Real-time applications, voice assistants, monitoring

**Behavior**:
- Runs calibration (30 samples, ~3 seconds)
- Starts background audio capture thread
- Processes audio continuously
- Logs voice detections in real-time
- Runs until Ctrl+C

**Sample Output**:
```
2024-01-15 10:30:15,123 - __main__ - INFO - Loading Silero VAD model...
2024-01-15 10:30:16,456 - __main__ - INFO - Silero VAD model loaded successfully
2024-01-15 10:30:16,457 - __main__ - INFO - Starting continuous audio processing...
2024-01-15 10:30:16,458 - __main__ - INFO - Calibrating noise profile with 30 samples...
2024-01-15 10:30:16,459 - __main__ - INFO - Please stay quiet during calibration...
2024-01-15 10:30:19,678 - __main__ - INFO - Average quiet energy: 145.23
2024-01-15 10:30:19,679 - __main__ - INFO - Listening for voice... (Press Ctrl+C to stop)
2024-01-15 10:30:25,123 - __main__ - INFO - 🎤 Voice detected! Probability: 0.87, Energy: 1234
2024-01-15 10:30:26,234 - __main__ - INFO - 🎤 Voice detected! Probability: 0.92, Energy: 1567
```

**Usage Tips**:
- Stay quiet during calibration for best results
- Speak normally after "Listening for voice..." message
- System adapts to your current environment
- Press Ctrl+C to stop

#### Mode 2: Single-Shot Processing

**Best for**: Testing, debugging, step-by-step analysis

**Behavior**:
- Captures one audio chunk at a time
- Processes immediately
- Shows detailed results for each chunk
- No background threading

**Sample Output**:
```
2024-01-15 10:35:15,123 - __main__ - INFO - Starting single-shot audio processing...
2024-01-15 10:35:15,124 - __main__ - INFO - Listening for voice... (Press Ctrl+C to stop)
2024-01-15 10:35:18,456 - __main__ - INFO - 🎤 Voice detected! Probability: 0.78, Energy: 987
2024-01-15 10:35:18,556 - __main__ - DEBUG - Audio detected (not voice): Probability: 0.23, Energy: 234
```

**Usage Tips**:
- Good for understanding system behavior
- Shows both voice and non-voice detections
- Useful for threshold tuning

#### Mode 3: Test Model

**Best for**: Installation verification, troubleshooting

**Behavior**:
- Captures single audio chunk
- Tests complete processing pipeline
- Reports success/failure
- Exits immediately

**Sample Output**:
```
2024-01-15 10:40:15,123 - __main__ - INFO - Testing Silero VAD model...
2024-01-15 10:40:15,456 - __main__ - INFO - Test results: Voice=True, Probability=0.85, Energy=1123
2024-01-15 10:40:15,457 - __main__ - INFO - Model test successful!
```


## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Issues

**Issue**: `Failed to load Silero VAD model`
```
ERROR:vad:Failed to load Silero VAD model: HTTP Error 403: Forbidden
```

**Causes & Solutions**:
- **Internet connectivity**: Check network connection
- **Firewall/proxy**: Configure firewall to allow PyTorch Hub access
- **GitHub rate limiting**: Wait and retry later
- **Cached corruption**: Clear cache and retry
  ```bash
  rm -rf ~/.cache/torch/hub/snakers4_silero-vad_master
  python main.py
  ```

**Issue**: `ModuleNotFoundError: No module named 'torch'`

**Solutions**:
```bash
# Verify virtual environment activation
which python  # Should point to your virtual environment

# Reinstall PyTorch
pip install torch torchaudio

# Check installation
python -c "import torch; print(torch.__version__)"
```

#### 2. Audio Device Issues

**Issue**: `OSError: [Errno -9996] Invalid input device`

**Diagnosis**:
```python
import pyaudio
p = pyaudio.PyAudio()
print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']} - {info['maxInputChannels']} input channels")
p.terminate()
```

**Solutions**:
- **macOS**: Grant microphone permission in System Preferences
- **Linux**: Check ALSA/PulseAudio configuration
- **Windows**: Verify microphone drivers and permissions
- **All platforms**: Use specific device index:
  ```python
  # In audio_capture.py
  stream = self.p.open(
      input_device_index=1,  # Use specific device
      # ... other parameters
  )
  ```

**Issue**: `No audio input detected`

**Diagnosis steps**:
```bash
# Test microphone
python -c "
import pyaudio
import numpy as np
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
data = stream.read(1024)
audio = np.frombuffer(data, dtype=np.int16)
print(f'Audio level: {np.mean(np.abs(audio))}')
stream.close()
p.terminate()
"
```

**Solutions**:
- Increase microphone gain/volume
- Check microphone connection
- Test with different audio applications
- Verify audio device permissions

## Key Achievements

1. **High Accuracy**: Uses state-of-the-art Silero VAD model for speech-specific detection
2. **Real-time Performance**: Processes audio with <50ms latency suitable for interactive applications
3. **Robust Architecture**: Handles errors gracefully with comprehensive fallback mechanisms
4. **Easy Integration**: Clean API allows simple integration into larger voice-processing pipelines
5. **Cross-platform Compatibility**: Works reliably across macOS, Linux, and Windows

---
## Wake Word Detection Approaches

### 1. Porcupine 
**Type**: Commercial solution  
**Cost**: Free (3 keywords) / $5/month (unlimited)  
**Accuracy**: High  
**Setup**: Easy  
**Offline**: Yes (after initialization)  

### 2. Template Matching
**Type**: Signal processing  
**Cost**: Free  
**Accuracy**: Medium  
**Setup**: Medium (record examples)  
**Offline**: Yes (completely)  
 

### 3. OpenWakeWord
**Type**: Open source ML  
**Cost**: Free  
**Accuracy**: High  
**Setup**: Complex ,license restriction applies to ALL pre-trained models (hey jarvis", "alexa",)
**Offline**: Yes  (after downloading)

---

## Command Detection Approaches

### Machine Learning Options

#### 1. Transfer Learning
Take existing trained model, adapt it for specific commands
How it works: Use pretrained model, replace last layer with your commands

**Complexity**: Medium  
**Data needed**: 50-100 examples per command  
**time**: 1-2 weeks  
**Accuracy**: High  
 

#### 2. Convolutional Neural Networks (CNN)
**Complexity**: High  
**Data needed**: 1000+ examples per command  
**time**: Weeks to months  
**Accuracy**: Very high  
Pros: ✅ Very accurate, ✅ Good with noise, ✅ Proven approach
Cons: ❌ Needs 1000+ examples per command, ❌ Complex to train

#### 3. Few-Shot Learning
ML that learns from just a few examples (5-10 per command)
How it works: Uses pre-trained models and adapts them quickly
**Complexity**: Medium  
**Data needed**: 5-10 examples per command  
**time**: 1 week  
**Accuracy**: High   
Pros: ✅ Needs few examples, ✅ Fast training, ✅ Good accuracy
Cons: ❌ Still complex setup,

--- 

## My Approach

### Option 1 :
Audio → Silero VAD → Whisper STT → Parse text for wake word + command
### Option 2 : 
Audio → Silero VAD → OpenWakeWord → Whisper STT → Parse command only

### Efficiency Analysis:
### Option 1 (Silero + Whisper):

Silent periods: 5ms Silero only 
Speech detected: 5ms Silero + 2000ms Whisper 
Problem: Whisper runs on ALL speech (even random talking)

### Option 2 (Silero + OpenWakeWord + Whisper):

Silent periods: 5ms Silero only 
Speech detected: 5ms Silero + 50ms OpenWakeWord 
Wake word detected: + 2000ms Whisper 
Advantage: Whisper only runs after wake word

### Performance Breakdown:

| Metric                      | Option 1 (Silero + Whisper)     | Option 2 (Silero + OpenWakeWord + Whisper) |
|----------------------------|----------------------------------|--------------------------------------------|
| **CPU when silent**        | Very Low (5ms checks)           | Very Low (5ms checks)                      |
| **CPU during random speech** | Very High (2000ms)             | Low (55ms)                                 |
| **CPU during wake word**   | Very High (2000ms)              | Very High (2055ms)                         |
| **False wake words**       | Higher                          | Lower                                      |
| **Power consumption**      | High                            | Low                                        |

---

## Wake word detection (Picovoice)

### 1. **Setting Up Picovoice Account and Console**

The first step was to sign up for a **Picovoice** account at the [Picovoice Console](https://console.picovoice.ai/). After creating the account, the following steps were performed:

- **Created a Custom Wake Word**: The wake word (e.g., "Hey Assistant") was selected and trained using Picovoice's **Porcupine** section in the console.
- **Downloaded the `.ppn` model file** for the wake word to be used locally within the ARS.

### 2. **Installing Required Libraries**

To integrate Porcupine into the project, the following Python libraries were installed:

```bash
pip install pvporcupine pyaudio
```

- `pvporcupine`: The core library for wake word detection.
- `pyaudio`: A Python library to capture audio input from a microphone.

### 3. **Training and Downloading the Custom Wake Word Model**

Using Picovoice Console:
- The custom wake word model was trained using Picovoice's **no-code interface**.
- The model was downloaded in the form of a `.ppn` file (platform-dependent, such as `linux_x86_64`).

### 4. **Integrating Wake Word Detection in the ARS**

The wake word detection was added to the ARS as follows:

- **Model Loading**: The custom `.ppn` model file was loaded using the `pvporcupine` library, specifying the path to the downloaded model.
- **Audio Capture**: The microphone input was captured using the `pyaudio` library.
- **Wake Word Detection**: The Porcupine engine continuously listened for the wake word and triggered a response when the wake word was detected.

Example code for integrating the wake word detection:



### 5. **Testing the System**

The ARS system was run, and Option 4 was selected to test the wake word detection functionality. Upon detection of the wake word, the system was set to trigger an appropriate response.


## Benefits of Using Picovoice Porcupine

1. **Offline Operation**: 
   - Once the wake word model is downloaded, it can be used entirely offline, ensuring privacy and reducing the need for constant internet connectivity.

2. **Low Latency**:
   - Porcupine is optimized for low-latency, making it ideal for real-time applications where quick response times are critical.

3. **Customization**:
   - You can easily create custom wake words tailored to your application, which provides flexibility in the design and functionality of your ARS.

4. **Cross-Platform**:
   - Porcupine supports multiple platforms, including Linux, macOS, Windows, Raspberry Pi, Android, and iOS, making it versatile for different hardware environments.

5. **Energy-Efficient**:
   - The Porcupine engine is designed to be resource-efficient, which is beneficial for embedded systems and mobile devices with limited processing power.

6. **No Internet Required (Post Setup)**:
   - Once the model is trained and downloaded, the system operates without needing an internet connection, providing flexibility in deployment.

## Limitations of Picovoice Porcupine

1. **Model Expiry**:
   - Custom models are valid for 30 days, and after this period, re-training is required. This is primarily relevant for cloud-based deployments, but offline models will still work indefinitely after the initial download.

2. **Limited Language Support**:
   - While Picovoice offers multiple languages, there may be limitations for certain languages or specific accents that could require additional training or model adjustment.

3. **Resource Usage**:
   - Although Porcupine is designed to be lightweight, performance may vary depending on the target device's processing power. High-resource environments (e.g., servers) may not face this issue, but embedded systems may require optimization.

4. **Training Model**:
   - The initial model training and download require an internet connection. Additionally, training a custom model might be limited to certain languages unless additional data is provided for training.

5. **AccessKey Usage**:
   - While the AccessKey allows for authentication and authorization, it may be subject to usage limits based on the plan chosen. For commercial applications, a paid plan may be required once your usage exceeds the free tier.

## Conclusion

By integrating Picovoice Porcupine, the Audio-Based Recognition System can now offer robust wake word detection capabilities, functioning offline and providing a responsive and customizable user experience. This integration enhances privacy, reduces cloud dependency, and ensures that the system works seamlessly across different platforms.

