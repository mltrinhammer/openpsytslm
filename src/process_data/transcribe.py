"""
Multi-GPU Language preprocessing pipeline: diarization (pyannote) and ASR (faster-whisper).

This script supports:
  - Multi-GPU setups with automatic load balancing
  - Distributed processing of audio files across available GPUs
  - No ffmpeg dependency (uses torchaudio/soundfile for audio I/O)
  - Simple speaker labeling (speaker_1, speaker_2)

Usage:
    # Auto-detect all GPUs
    python transcribe.py --input_dir /path/to/audio --output_dir /path/to/output
    
    # Use specific GPUs
    python transcribe.py --input_dir /path --output_dir /out --gpus 0,1,2,3
    
    # CPU-only mode
    python transcribe.py --input_dir /path --output_dir /out --cpu

Required packages:
   
    For pyannote.audio, you must also:
    1. Create a HuggingFace account at https://huggingface.co
    2. Accept the model terms at:
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0
    3. Get your token at https://huggingface.co/settings/tokens
    4. Set HF_TOKEN in config file or use --hf_token argument
"""
#UV specific features as this script has other package dependencies than root .venv
#https://docs.astral.sh/uv/guides/scripts/#running-a-script-with-dependencies

# /// script
# dependencies = [
#     "faster-whisper==1.2.0",
#     "ffmpeg-python==0.2.0",
#     "huggingface-hub==0.35.3",
#     "moviepy",
#     "psutil",
#     "openai-whisper==20250625",
#     "optuna==4.5.0",
#     "pandas==2.3.3",
#     "psutil==7.1.0",
#     "pyannote.audio==3.4.0",
#     "pydub==0.25.1",
#     "scikit-learn==1.7.2",
#     "speechbrain==1.0.3",
#     "torch==2.5.1+cu121",
#     "torchaudio==2.5.1+cu121",
#     "torchvision==0.20.1+cu121",
#     "transformers==4.56.2",
# ]
# 
# [tool.uv]
# index = [
#     { url = "https://download.pytorch.org/whl/cu121" },
# ]
# ///

import argparse
import json
import time
import io
import os
import platform
import sys
from pathlib import Path
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import torchaudio
import numpy as np

# psutil is optional but provides better system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[transcribe] Warning: psutil not installed. Install with 'pip install psutil' for better system monitoring.")

from utils import load_config


# ============================================================================
# Dependency Checks
# ============================================================================

def check_dependencies():
    """Check if required dependencies are installed and provide helpful error messages."""
    missing = []
    version_issues = []
    
    # Print versions first
    print(f"[transcribe] torch version: {torch.__version__}")
    print(f"[transcribe] torchaudio version: {torchaudio.__version__}")
    
    # Check pyannote.audio
    try:
        import pyannote.audio
        print(f"[transcribe] ✓ pyannote.audio version: {pyannote.audio.__version__}")
    except ImportError:
        missing.append(("pyannote.audio", "uv pip install pyannote.audio"))
    except AttributeError as e:
        if "AudioMetaData" in str(e):
            version_issues.append((
                "pyannote.audio / torchaudio version mismatch",
                f"Your torchaudio ({torchaudio.__version__}) is incompatible with pyannote.audio.\n"
                f"  Fix with ONE of:\n"
                f"    uv pip install --upgrade pyannote.audio   # upgrade pyannote\n"
                f"    uv pip install 'torchaudio>=2.0.0,<2.1.0' # or downgrade torchaudio"
            ))
        else:
            raise
    
    # Check faster-whisper
    try:
        import faster_whisper
        print(f"[transcribe] ✓ faster-whisper installed")
    except ImportError:
        missing.append(("faster-whisper", "uv pip install faster-whisper"))
    
    # Check moviepy (optional, for video files)
    try:
        import moviepy
        print(f"[transcribe] ✓ moviepy installed")
    except ImportError:
        print("[transcribe] ⚠ moviepy not installed (only needed for video files)")
    
    if version_issues:
        print("\n" + "=" * 70)
        print("ERROR: Package version incompatibility")
        print("=" * 70)
        for issue, fix in version_issues:
            print(f"\n  Issue: {issue}")
            print(f"  {fix}")
        print("\n" + "=" * 70 + "\n")
        return False
    
    if missing:
        print("\n" + "=" * 70)
        print("ERROR: Missing required dependencies")
        print("=" * 70)
        for pkg, install_cmd in missing:
            print(f"\n  Missing: {pkg}")
            print(f"  Install: {install_cmd}")
        
        print("\n" + "-" * 70)
        print("For pyannote.audio, you must also accept the model terms:")
        print("  1. https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  2. https://huggingface.co/pyannote/segmentation-3.0")
        print("\nThen set your HuggingFace token:")
        print("  export HF_TOKEN='your_token_here'")
        print("  # or use --hf_token argument")
        print("=" * 70 + "\n")
        
        return False
    
    return True


# ============================================================================
# System & Environment Logging (for HPC/SLURM)
# ============================================================================

def get_system_memory() -> dict:
    """Get system memory info, with fallback if psutil not available."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_gb": mem.used / 1e9,
            "percent": mem.percent
        }
    else:
        # Fallback: try to read /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            meminfo = {}
            for line in lines:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val) * 1024  # Convert from KB to bytes
            
            total = meminfo.get('MemTotal', 0) / 1e9
            free = meminfo.get('MemFree', 0) / 1e9
            available = meminfo.get('MemAvailable', free) / 1e9
            used = total - available
            return {
                "total_gb": total,
                "available_gb": available,
                "used_gb": used,
                "percent": (used / total * 100) if total > 0 else 0
            }
        except Exception:
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0}


def get_cpu_info() -> dict:
    """Get CPU info, with fallback if psutil not available."""
    if HAS_PSUTIL:
        return {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=0.5)
        }
    else:
        # Fallback
        try:
            import os
            cores = os.cpu_count() or 1
            return {
                "physical_cores": cores,
                "logical_cores": cores,
                "usage_percent": 0
            }
        except Exception:
            return {"physical_cores": 1, "logical_cores": 1, "usage_percent": 0}


def print_system_info():
    """Print comprehensive system information for SLURM logs."""
    print("\n" + "=" * 70)
    print("SYSTEM & ENVIRONMENT INFORMATION")
    print("=" * 70)
    
    # Timestamp
    print(f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Host info
    print(f"\n--- Host Information ---")
    print(f"  Hostname:     {platform.node()}")
    print(f"  Platform:     {platform.platform()}")
    print(f"  Python:       {platform.python_version()}")
    
    # SLURM environment
    slurm_vars = {
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "N/A"),
        "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME", "N/A"),
        "SLURM_NODELIST": os.environ.get("SLURM_NODELIST", "N/A"),
        "SLURM_NTASKS": os.environ.get("SLURM_NTASKS", "N/A"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK", "N/A"),
        "SLURM_MEM_PER_NODE": os.environ.get("SLURM_MEM_PER_NODE", "N/A"),
        "SLURM_GPUS": os.environ.get("SLURM_GPUS", "N/A"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS", "N/A"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
    }
    
    print(f"\n--- SLURM Environment ---")
    for key, val in slurm_vars.items():
        print(f"  {key}: {val}")
    
    # CPU info
    cpu_info = get_cpu_info()
    print(f"\n--- CPU Information ---")
    print(f"  Physical cores: {cpu_info['physical_cores']}")
    print(f"  Logical cores:  {cpu_info['logical_cores']}")
    print(f"  CPU usage:      {cpu_info['usage_percent']:.1f}%")
    
    # RAM info
    mem = get_system_memory()
    print(f"\n--- System Memory ---")
    print(f"  Total:     {mem['total_gb']:.1f} GB")
    print(f"  Available: {mem['available_gb']:.1f} GB")
    print(f"  Used:      {mem['used_gb']:.1f} GB ({mem['percent']:.1f}%)")
    
    # PyTorch & CUDA info
    print(f"\n--- PyTorch & CUDA ---")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version:    {torch.version.cuda}")
        print(f"  cuDNN version:   {torch.backends.cudnn.version()}")
        print(f"  GPU count:       {torch.cuda.device_count()}")
        
        print(f"\n--- GPU Details ---")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory:       {props.total_memory / 1e9:.2f} GB")
            print(f"    Free memory:        {free / 1e9:.2f} GB")
            print(f"    Multiprocessors:    {props.multi_processor_count}")
    else:
        print("  No CUDA GPUs available")
    
    print("\n" + "=" * 70 + "\n")


def print_job_summary(config, gpus: List[int], num_files: int, workers_per_gpu: int):
    """Print job configuration summary."""
    print("\n" + "=" * 70)
    print("JOB CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\n--- Processing Configuration ---")
    print(f"  Files to process:    {num_files}")
    print(f"  GPUs:                {gpus if gpus else 'CPU only'}")
    print(f"  Workers per GPU:     {workers_per_gpu}")
    print(f"  Total workers:       {len(gpus) * workers_per_gpu if gpus else workers_per_gpu}")
    
    print(f"\n--- Model Configuration ---")
    print(f"  Diarization model:   {config.diary_model}")
    print(f"  Num speakers:        {config.diarization_num_speakers}")
    print(f"  Whisper model:       {config.whisper_model_size}")
    print(f"  Whisper language:    {config.whisper_language}")
    print(f"  Compute type:        {config.whisper_compute_type}")
    print(f"  Beam size:           {config.whisper_beam_size}")
    print(f"  Batch size:          {config.whisper_batch_size}")
    
    print(f"\n--- Audio Configuration ---")
    print(f"  Sample rate:         {config.audio_sample_rate} Hz")
    print(f"  Min segment:         {config.min_segment_ms} ms")
    print(f"  Both speakers:       {config.both_speakers}")
    
    print("\n" + "=" * 70 + "\n")


def print_periodic_status(gpus: List[int], start_time: float, completed: int, total: int):
    """Print periodic status update with GPU memory."""
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    
    timestamp = time.strftime("%H:%M:%S")
    
    print(f"\n{'─'*70}")
    print(f"[{timestamp}] PROGRESS: {completed}/{total} files ({completed/total*100:.1f}%)")
    print(f"           Elapsed: {elapsed/60:.1f} min | Rate: {rate*60:.1f} files/min | ETA: {eta/60:.1f} min")
    
    # System memory
    mem = get_system_memory()
    print(f"           System RAM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB ({mem['percent']:.0f}%)")
    
    # GPU memory
    if gpus:
        for idx in gpus:
            info = get_gpu_utilization(idx)
            bar = format_memory_bar(info['memory_percent'], width=20)
            print(f"           GPU {idx}: {bar} ({info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f} GB)")
    
    print(f"{'─'*70}\n")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TranscribeConfig:
    """Configuration for transcription pipeline."""
    # Diarization
    diary_model: str = "pyannote/speaker-diarization-3.1"
    diarization_num_speakers: int = 2
    hf_token: Optional[str] = None
    
    # ASR
    whisper_model_size: str = "small"
    whisper_compute_type: str = "float16"  # float16 for GPU, int8 for CPU
    whisper_language: Optional[str] = None  # REQUIRED: set in config_language.yaml
    whisper_beam_size: int = 3
    whisper_batch_size: int = 8
    
    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    min_segment_ms: int = 1500  # Minimum segment length to transcribe
    
    # Processing
    both_speakers: bool = True  # Transcribe both speakers
    
    @classmethod
    def from_dict(cls, d: dict) -> "TranscribeConfig":
        """Create config from dictionary, using defaults for missing keys."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def validate(self):
        """Validate required config values."""
        if not self.whisper_language:
            raise ValueError(
                "whisper_language must be set in config_language.yaml "
                "(e.g., whisper_language: 'en' or 'de')"
            )


# ============================================================================
# GPU Management
# ============================================================================

def get_available_gpus() -> List[int]:
    """Get list of available CUDA GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_gpu_memory_info(device_idx: int) -> Tuple[float, float, float]:
    """Get GPU memory info in GB: (used, free, total)."""
    try:
        free, total = torch.cuda.mem_get_info(device_idx)
        used = total - free
        return used / 1e9, free / 1e9, total / 1e9
    except Exception:
        return 0.0, 0.0, 0.0


def get_gpu_utilization(device_idx: int) -> dict:
    """Get comprehensive GPU info including utilization estimates."""
    info = {
        "device_idx": device_idx,
        "name": "Unknown",
        "memory_used_gb": 0.0,
        "memory_free_gb": 0.0,
        "memory_total_gb": 0.0,
        "memory_percent": 0.0,
        "compute_capability": "Unknown",
    }
    
    try:
        if not torch.cuda.is_available():
            return info
            
        info["name"] = torch.cuda.get_device_name(device_idx)
        
        # Memory info
        free, total = torch.cuda.mem_get_info(device_idx)
        used = total - free
        info["memory_used_gb"] = used / 1e9
        info["memory_free_gb"] = free / 1e9
        info["memory_total_gb"] = total / 1e9
        info["memory_percent"] = (used / total) * 100 if total > 0 else 0
        
        # Compute capability
        major, minor = torch.cuda.get_device_capability(device_idx)
        info["compute_capability"] = f"{major}.{minor}"
        
        # PyTorch memory tracking (more detailed)
        if torch.cuda.is_initialized():
            info["pytorch_allocated_gb"] = torch.cuda.memory_allocated(device_idx) / 1e9
            info["pytorch_reserved_gb"] = torch.cuda.memory_reserved(device_idx) / 1e9
            info["pytorch_max_allocated_gb"] = torch.cuda.max_memory_allocated(device_idx) / 1e9
        
    except Exception as e:
        info["error"] = str(e)
    
    return info


def format_memory_bar(percent: float, width: int = 30) -> str:
    """Create a visual memory usage bar."""
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent:5.1f}%"


def print_gpu_status(gpus: List[int], prefix: str = "[transcribe]", detailed: bool = False):
    """Print memory status for all GPUs with visual formatting."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*70}")
    print(f"{prefix} GPU STATUS REPORT - {timestamp}")
    print(f"{'='*70}")
    
    for idx in gpus:
        info = get_gpu_utilization(idx)
        
        print(f"\n  GPU {idx}: {info['name']}")
        print(f"  {'─'*50}")
        print(f"  Memory:  {format_memory_bar(info['memory_percent'])}")
        print(f"           {info['memory_used_gb']:.2f} GB / {info['memory_total_gb']:.2f} GB "
              f"(Free: {info['memory_free_gb']:.2f} GB)")
        
        if detailed and "pytorch_allocated_gb" in info:
            print(f"  PyTorch: Allocated={info['pytorch_allocated_gb']:.2f} GB, "
                  f"Reserved={info['pytorch_reserved_gb']:.2f} GB, "
                  f"Peak={info['pytorch_max_allocated_gb']:.2f} GB")
        
        print(f"  Compute: Capability {info['compute_capability']}")
    
    print(f"{'='*70}\n")


def log_gpu_event(device: str, event: str, file_name: str = "", extra_info: dict = None):
    """Log a GPU-related event with timestamp and memory info."""
    timestamp = time.strftime("%H:%M:%S")
    
    # Extract device index
    device_idx = 0
    if "cuda:" in device:
        try:
            device_idx = int(device.split(":")[1])
        except ValueError:
            pass
    elif device == "cpu":
        print(f"[{timestamp}] [{device}] {event} | {file_name}")
        return
    
    info = get_gpu_utilization(device_idx)
    mem_str = f"Mem: {info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f}GB ({info['memory_percent']:.0f}%)"
    
    extra_str = ""
    if extra_info:
        extra_str = " | " + ", ".join(f"{k}={v}" for k, v in extra_info.items())
    
    file_str = f" | {file_name}" if file_name else ""
    print(f"[{timestamp}] [GPU {device_idx}] {event}{file_str} | {mem_str}{extra_str}", flush=True)


# ============================================================================
# Audio I/O - Simple and Robust using ffmpeg-python
# ============================================================================

def load_audio(file_path: Path, target_sr: int = 16000, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio from any file format using ffmpeg (handles video files too).
    
    This is the most reliable method as ffmpeg supports virtually all formats.
    
    Args:
        file_path: Path to audio/video file
        target_sr: Target sample rate (default 16000)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (waveform tensor [channels, samples], sample_rate)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"[transcribe] Loading audio: {file_path.name}")
    
    # Method 1: Try ffmpeg-python (most reliable for any format)
    try:
        waveform, sr = _load_with_ffmpeg(file_path, target_sr, mono)
        return waveform, sr
    except Exception as e1:
        print(f"[transcribe] ffmpeg-python failed: {e1}")
    
    # Method 2: Try pydub as fallback
    try:
        waveform, sr = _load_with_pydub(file_path, target_sr, mono)
        return waveform, sr
    except Exception as e2:
        print(f"[transcribe] pydub failed: {e2}")
    
    # Method 3: Try torchaudio for simple audio files
    try:
        waveform, sr = torchaudio.load(str(file_path))
        waveform, sr = _process_waveform(waveform, sr, target_sr, mono)
        print(f"[transcribe] Loaded with torchaudio: {waveform.shape[1]/sr:.1f}s")
        return waveform, sr
    except Exception as e3:
        print(f"[transcribe] torchaudio failed: {e3}")
    
    # All methods failed
    raise RuntimeError(
        f"Failed to load audio from {file_path.name}\n\n"
        f"Errors:\n"
        f"  ffmpeg: {e1}\n"
        f"  pydub: {e2}\n"
        f"  torchaudio: {e3}\n\n"
        f"Try manually converting first:\n"
        f"  ffmpeg -i '{file_path.name}' -ar 16000 -ac 1 '{file_path.stem}.wav'"
    )


def _load_with_ffmpeg(file_path: Path, target_sr: int, mono: bool) -> Tuple[torch.Tensor, int]:
    """Load audio using ffmpeg-python - works with any format including video."""
    import ffmpeg
    import subprocess
    
    print(f"[transcribe] Using ffmpeg to extract audio...")
    
    channels = 1 if mono else 2
    
    try:
        # Use ffmpeg to extract audio as raw PCM
        out, err = (
            ffmpeg
            .input(str(file_path))
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=channels, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(out, np.float32)
        
        if audio_array.size == 0:
            raise RuntimeError("ffmpeg returned empty audio")
        
        # Reshape based on channels
        if channels == 1:
            waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
        else:
            # Reshape to (samples, channels) then transpose to (channels, samples)
            audio_array = audio_array.reshape(-1, channels)
            waveform = torch.from_numpy(audio_array.T).float()
        
        duration = waveform.shape[1] / target_sr
        print(f"[transcribe] ffmpeg extracted: {duration:.1f}s, {waveform.shape[0]} channel(s)")
        
        return waveform, target_sr
        
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else 'Unknown error'
        raise RuntimeError(f"ffmpeg error: {stderr}")


def _load_with_pydub(file_path: Path, target_sr: int, mono: bool) -> Tuple[torch.Tensor, int]:
    """Load audio using pydub - good fallback for various formats."""
    from pydub import AudioSegment
    
    print(f"[transcribe] Using pydub to load audio...")
    
    # Load audio file
    audio = AudioSegment.from_file(str(file_path))
    
    # Convert to mono if needed
    if mono:
        audio = audio.set_channels(1)
    
    # Resample if needed
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize to float32 [-1, 1]
    if audio.sample_width == 2:  # 16-bit
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32) / np.max(np.abs(samples))
    
    # Reshape for stereo if needed
    if audio.channels == 2 and not mono:
        samples = samples.reshape(-1, 2).T
        waveform = torch.from_numpy(samples).float()
    else:
        waveform = torch.from_numpy(samples).float().unsqueeze(0)
    
    duration = waveform.shape[1] / target_sr
    print(f"[transcribe] pydub loaded: {duration:.1f}s, {waveform.shape[0]} channel(s)")
    
    return waveform, target_sr


def _process_waveform(waveform: torch.Tensor, sr: int, target_sr: int, mono: bool) -> Tuple[torch.Tensor, int]:
    """Process loaded waveform: convert to mono and resample if needed."""
    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    return waveform, sr


def save_audio(waveform: torch.Tensor, file_path: Path, sample_rate: int = 16000):
    """Save waveform to WAV file using torchaudio."""
    torchaudio.save(str(file_path), waveform, sample_rate)


def extract_segment(waveform: torch.Tensor, sr: int, start_sec: float, end_sec: float) -> torch.Tensor:
    """Extract a segment from waveform without writing to disk."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return waveform[:, start_sample:end_sample]


# ============================================================================
# Diarization
# ============================================================================

def load_diarization_pipeline(config: TranscribeConfig, device: str):
    """Load pyannote diarization pipeline."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            f"pyannote.audio is not installed. Install with:\n"
            f"  pip install pyannote.audio\n\n"
            f"You must also accept the model terms at:\n"
            f"  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            f"  https://huggingface.co/pyannote/segmentation-3.0\n\n"
            f"Original error: {e}"
        )
    
    log_gpu_event(device, "Loading diarization model", config.diary_model)
    start = time.time()
    
    if not config.hf_token:
        print("[transcribe] WARNING: No HuggingFace token provided!")
        print("  Set HF_TOKEN environment variable or use --hf_token argument")
        print("  Get your token at: https://huggingface.co/settings/tokens")
    
    try:
        pipeline = Pipeline.from_pretrained(
            config.diary_model, 
            use_auth_token=config.hf_token
        )
    except Exception as e:
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise RuntimeError(
                f"HuggingFace authentication failed for {config.diary_model}.\n"
                f"Please ensure you have:\n"
                f"  1. A valid HuggingFace token (set HF_TOKEN or use --hf_token)\n"
                f"  2. Accepted the model terms at:\n"
                f"     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                f"     https://huggingface.co/pyannote/segmentation-3.0\n"
                f"Original error: {e}"
            )
        raise
    
    # Move to device
    device_obj = torch.device(device)
    try:
        pipeline = pipeline.to(device_obj)
    except Exception as e:
        print(f"[transcribe] Warning: Could not move diarization to {device}: {e}")
    
    elapsed = time.time() - start
    log_gpu_event(device, f"Diarization model loaded in {elapsed:.1f}s", config.diary_model)
    return pipeline


def run_diarization(
    audio_path: Path,
    output_dir: Path,
    config: TranscribeConfig,
    device: str = "cuda:0"
) -> Optional[Path]:
    """
    Run speaker diarization on audio file.
    
    Returns:
        Path to RTTM file, or None on failure
    """
    start = time.time()
    file_subdir = output_dir / audio_path.stem
    file_subdir.mkdir(exist_ok=True)
    
    rttm_path = file_subdir / f"{audio_path.stem}.rttm"
    
    log_gpu_event(device, "START diarization", audio_path.name)
    
    try:
        # First, load the audio using our robust load_audio function
        # This handles MP4, MOV, and other video formats that pyannote can't read directly
        log_gpu_event(device, "Loading audio file", audio_path.name)
        waveform, sample_rate = load_audio(audio_path, target_sr=16000, mono=False)
        
        # pyannote expects {"waveform": tensor, "sample_rate": int}
        # waveform shape should be (channel, time)
        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        log_gpu_event(device, f"Audio loaded: {waveform.shape[1]/sample_rate:.1f}s", audio_path.name)
        
        # Load the diarization pipeline
        pipeline = load_diarization_pipeline(config, device)
        
        log_gpu_event(device, "Running diarization inference", audio_path.name)
        
        # Run diarization with pre-loaded audio (not file path)
        diarization = pipeline(
            audio_input,
            num_speakers=config.diarization_num_speakers
        )
        
        # Sanitize URI for RTTM format (no spaces allowed)
        try:
            diarization.uri = re.sub(r"\s+", "_", audio_path.stem)
        except Exception:
            pass
        
        # Write RTTM
        with open(rttm_path, "w") as f:
            diarization.write_rttm(f)
        
        elapsed = time.time() - start
        log_gpu_event(device, f"COMPLETE diarization ({elapsed:.1f}s)", audio_path.name)
        return rttm_path
        
    except Exception as e:
        log_gpu_event(device, f"FAILED diarization: {e}", audio_path.name)
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# ASR (Faster-Whisper)
# ============================================================================

def load_whisper_model(config: TranscribeConfig, device: str):
    """Load faster-whisper model."""
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    device_index = 0
    if ":" in device:
        try:
            device_index = int(device.split(":")[1])
        except ValueError:
            pass
    
    # Adjust compute type for CPU
    compute_type = config.whisper_compute_type
    if device_type == "cpu" and compute_type == "float16":
        compute_type = "int8"
        print(f"[transcribe] Switching to int8 compute for CPU")
    
    log_gpu_event(device, f"Loading Whisper model '{config.whisper_model_size}'", 
                  extra_info={"compute_type": compute_type})
    
    start = time.time()
    model = WhisperModel(
        config.whisper_model_size,
        device=device_type,
        device_index=device_index,
        compute_type=compute_type,
    )
    batched = BatchedInferencePipeline(model=model)
    
    elapsed = time.time() - start
    log_gpu_event(device, f"Whisper model loaded in {elapsed:.1f}s", config.whisper_model_size)
    
    return batched


def parse_rttm(rttm_path: Path) -> List[dict]:
    """Parse RTTM file into list of segments with speaker labels."""
    from pyannote.core import Annotation, Segment
    
    annotation = Annotation()
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start_t = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segment = Segment(start_t, start_t + duration)
                annotation[segment] = speaker
    
    # Map speakers to speaker_1, speaker_2 (most speaking = speaker_1)
    labels = annotation.labels()
    if len(labels) >= 2:
        primary = annotation.argmax()
        secondary = [l for l in labels if l != primary][0]
        mapping = {primary: "speaker_1", secondary: "speaker_2"}
    elif len(labels) == 1:
        mapping = {labels[0]: "speaker_1"}
    else:
        mapping = {}
    
    segments = []
    for segment in annotation.itersegments():
        orig_label = next(iter(annotation.get_labels(segment)))
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "duration": segment.end - segment.start,
            "speaker": mapping.get(orig_label, "unknown")
        })
    
    return segments


def run_asr(
    audio_path: Path,
    output_dir: Path,
    config: TranscribeConfig,
    device: str = "cuda:0"
) -> Optional[Path]:
    """
    Run ASR on audio file using diarization segments.
    
    Returns:
        Path to results JSON, or None on failure
    """
    start = time.time()
    file_subdir = output_dir / audio_path.stem
    rttm_path = file_subdir / f"{audio_path.stem}.rttm"
    results_path = file_subdir / f"results_{audio_path.stem}.json"
    
    if not rttm_path.exists():
        print(f"[transcribe] No RTTM for {audio_path.name}. Run diarization first.")
        return None
    
    log_gpu_event(device, "START ASR", audio_path.name)
    
    try:
        # Load audio
        log_gpu_event(device, "Loading audio", audio_path.name)
        waveform, sr = load_audio(audio_path, target_sr=config.audio_sample_rate, mono=True)
        audio_duration_sec = waveform.shape[1] / sr
        log_gpu_event(device, f"Audio loaded ({audio_duration_sec:.1f}s)", audio_path.name)
        
        # Parse diarization
        segments = parse_rttm(rttm_path)
        log_gpu_event(device, f"Parsed {len(segments)} diarization segments", audio_path.name)
        
        # Load model
        batched_model = load_whisper_model(config, device)
        
        # Transcribe each segment
        results = []
        temp_chunk = file_subdir / "_temp_chunk.wav"
        
        segments_to_process = [
            seg for seg in segments
            if (config.both_speakers or seg["speaker"] == "speaker_1")
            and seg["duration"] * 1000 >= config.min_segment_ms
        ]
        
        log_gpu_event(device, f"Transcribing {len(segments_to_process)} segments", audio_path.name)
        
        for i, seg in enumerate(segments_to_process):
            # Log progress every 10 segments or at start/end
            if i == 0 or (i + 1) % 10 == 0 or i == len(segments_to_process) - 1:
                log_gpu_event(device, f"Segment {i+1}/{len(segments_to_process)}", audio_path.name,
                             extra_info={"speaker": seg["speaker"], "duration": f"{seg['duration']:.1f}s"})
            
            # Extract segment
            chunk_waveform = extract_segment(waveform, sr, seg["start"], seg["end"])
            
            # Save temp chunk for whisper (faster-whisper needs file path)
            save_audio(chunk_waveform, temp_chunk, sr)
            
            # Transcribe
            transcription_segments, _ = batched_model.transcribe(
                str(temp_chunk),
                beam_size=config.whisper_beam_size,
                language=config.whisper_language,
                condition_on_previous_text=False,
                word_timestamps=False,
                batch_size=config.whisper_batch_size,
            )
            
            for t in transcription_segments:
                results.append({
                    "text": t.text.strip(),
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"]
                })
        
        # Cleanup temp file
        if temp_chunk.exists():
            temp_chunk.unlink()
        
        # Save results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start
        log_gpu_event(device, f"COMPLETE ASR ({elapsed:.1f}s, {len(results)} results)", audio_path.name)
        return results_path
        
    except Exception as e:
        log_gpu_event(device, f"FAILED ASR: {e}", audio_path.name)
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Multi-GPU Processing
# ============================================================================

def process_file_on_device(
    audio_path: str,
    output_dir: str,
    config_dict: dict,
    device: str,
    stage: str
) -> dict:
    """
    Process a single file on specified device.
    
    This function is designed to be called in a separate process.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    config = TranscribeConfig.from_dict(config_dict)
    
    result = {
        "file": str(audio_path),
        "device": device,
        "status": "success",
        "outputs": []
    }
    
    try:
        if stage in ("diarization", "full"):
            rttm = run_diarization(audio_path, output_dir, config, device)
            if rttm:
                result["outputs"].append(str(rttm))
        
        if stage in ("asr", "full"):
            json_out = run_asr(audio_path, output_dir, config, device)
            if json_out:
                result["outputs"].append(str(json_out))
                
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def distribute_files_to_gpus(files: List[Path], gpus: List[int]) -> List[Tuple[Path, str]]:
    """
    Distribute files across GPUs in round-robin fashion.
    
    Returns list of (file_path, device_string) tuples.
    """
    assignments = []
    for i, f in enumerate(files):
        gpu_idx = gpus[i % len(gpus)]
        device = f"cuda:{gpu_idx}"
        assignments.append((f, device))
    return assignments


def process_files_parallel(
    files: List[Path],
    output_dir: Path,
    config: TranscribeConfig,
    gpus: List[int],
    stage: str,
    max_workers_per_gpu: int = 1
) -> List[dict]:
    """
    Process files in parallel across multiple GPUs.
    
    Args:
        files: List of audio file paths
        output_dir: Output directory
        config: Transcription config
        gpus: List of GPU indices to use
        stage: Pipeline stage (diarization, asr, full)
        max_workers_per_gpu: Max concurrent processes per GPU
    """
    if not gpus:
        # CPU-only mode
        gpus_or_cpu = ["cpu"]
    else:
        gpus_or_cpu = gpus
    
    total_workers = len(gpus_or_cpu) * max_workers_per_gpu
    
    print(f"\n{'─'*70}")
    print(f"[transcribe] Parallel Processing Configuration:")
    print(f"  Files to process: {len(files)}")
    print(f"  Devices:          {gpus_or_cpu}")
    print(f"  Workers/device:   {max_workers_per_gpu}")
    print(f"  Total workers:    {total_workers}")
    print(f"{'─'*70}\n")
    
    # Distribute files
    if gpus:
        assignments = distribute_files_to_gpus(files, gpus)
        # Log file distribution
        for gpu_idx in gpus:
            gpu_files = [f.name for f, d in assignments if d == f"cuda:{gpu_idx}"]
            print(f"[transcribe] GPU {gpu_idx} assigned {len(gpu_files)} files")
    else:
        assignments = [(f, "cpu") for f in files]
    
    config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
    
    results = []
    start_time = time.time()
    status_interval = 5  # Print status every N completed files
    
    # Use spawn to avoid CUDA context issues across processes
    ctx = mp.get_context("spawn")
    
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=ctx) as executor:
        futures = {}
        for audio_path, device in assignments:
            future = executor.submit(
                process_file_on_device,
                str(audio_path),
                str(output_dir),
                config_dict,
                device,
                stage
            )
            futures[future] = audio_path
        
        completed = 0
        for future in as_completed(futures):
            audio_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                # Log completion
                status = "✓" if result["status"] == "success" else "✗"
                print(f"[{time.strftime('%H:%M:%S')}] {status} Completed: {audio_path.name}", flush=True)
                
            except Exception as e:
                results.append({
                    "file": str(audio_path),
                    "status": "error",
                    "error": str(e)
                })
                print(f"[{time.strftime('%H:%M:%S')}] ✗ Failed: {audio_path.name} - {e}", flush=True)
            
            completed += 1
            
            # Periodic status update
            if completed % status_interval == 0 or completed == len(files):
                print_periodic_status(gpus, start_time, completed, len(files))
    
    return results


# ============================================================================
# Main
# ============================================================================

def discover_audio_files(input_dir: Path, temp_audio_dir: Optional[Path] = None) -> List[Path]:
    """Discover audio files recursively (WAV only)."""
    extensions = ["*.wav"]
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(ext))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU transcription pipeline (diarization + ASR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all GPUs
  python transcribe.py --input_dir ./audio --output_dir ./output
  
  # Use specific GPUs
  python transcribe.py --input_dir ./audio --output_dir ./output --gpus 0,1,2,3
  
  # CPU-only mode
  python transcribe.py --input_dir ./audio --output_dir ./output --cpu
  
  # Run only ASR (diarization already done)
  python transcribe.py --input_dir ./audio --output_dir ./output --stage asr
        """
    )
    parser.add_argument("--config", type=str, default="config_language.yaml",
                        help="Path to config file")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing audio/video files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--stage", type=str, choices=["diarization", "asr", "full"],
                        default="full", help="Pipeline stage to run")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (e.g., '0,1,2,3'). Default: all available")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Max concurrent workers per GPU (default: 1)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files with existing outputs")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for pyannote (or set HF_TOKEN env var)")
    parser.add_argument("--check_deps", action="store_true",
                        help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    # =========================================================================
    # Check dependencies first
    # =========================================================================
    if not check_dependencies():
        print("[transcribe] Exiting due to missing dependencies")
        sys.exit(1)
    
    if args.check_deps:
        print("[transcribe] All dependencies are installed!")
        sys.exit(0)
    
    # =========================================================================
    # Print comprehensive system info for SLURM logs
    # =========================================================================
    print_system_info()
    
    # Load config
    config_dict = load_config(args.config) or {}
    
    # Override with CLI args
    if args.hf_token:
        config_dict["hf_token"] = args.hf_token
    elif os.environ.get("HF_TOKEN"):
        config_dict["hf_token"] = os.environ["HF_TOKEN"]
    
    config = TranscribeConfig.from_dict(config_dict)
    
    # Validate required config values
    config.validate()
    
    # Determine GPUs
    if args.cpu:
        gpus = []
        print("[transcribe] Running in CPU-only mode")
    elif args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        print(f"[transcribe] Using specified GPUs: {gpus}")
    else:
        gpus = get_available_gpus()
        if gpus:
            print(f"[transcribe] Auto-detected {len(gpus)} GPU(s): {gpus}")
        else:
            print("[transcribe] No GPUs detected, using CPU")
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover files
    audio_files = discover_audio_files(input_dir)
    
    if not audio_files:
        print(f"[transcribe] No audio/video files found in {input_dir}")
        return
    
    # Filter existing if requested
    if args.skip_existing:
        filtered = []
        for f in audio_files:
            file_subdir = output_dir / f.stem
            rttm = file_subdir / f"{f.stem}.rttm"
            results = file_subdir / f"results_{f.stem}.json"
            
            if args.stage == "diarization" and rttm.exists():
                print(f"[transcribe] Skip (RTTM exists): {f.name}")
                continue
            if args.stage == "asr" and results.exists():
                print(f"[transcribe] Skip (results exist): {f.name}")
                continue
            if args.stage == "full" and rttm.exists() and results.exists():
                print(f"[transcribe] Skip (all outputs exist): {f.name}")
                continue
            
            filtered.append(f)
        
        audio_files = filtered
    
    if not audio_files:
        print("[transcribe] Nothing to process")
        return
    
    # =========================================================================
    # Print job configuration summary
    # =========================================================================
    print_job_summary(config, gpus, len(audio_files), args.workers_per_gpu)
    
    # Print initial GPU status with detailed memory info
    if gpus:
        print_gpu_status(gpus, detailed=True)
    
    # Process files
    print(f"\n[transcribe] Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[transcribe] Stage: {args.stage}")
    print("-" * 70)
    
    start = time.time()
    results = process_files_parallel(
        audio_files,
        output_dir,
        config,
        gpus,
        args.stage,
        max_workers_per_gpu=args.workers_per_gpu
    )
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    elapsed = time.time() - start
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print("\n" + "=" * 70)
    print("FINAL JOB SUMMARY")
    print("=" * 70)
    print(f"\n  Completed at:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time:      {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"  Files processed: {len(results)}")
    print(f"  Successful:      {success}")
    print(f"  Failed:          {errors}")
    
    if len(results) > 0 and elapsed > 0:
        print(f"\n  Throughput:      {len(results)/elapsed*60:.2f} files/minute")
        print(f"  Avg time/file:   {elapsed/len(results):.1f} seconds")
    
    # Final GPU status
    if gpus:
        print(f"\n--- Final GPU Memory Status ---")
        for idx in gpus:
            info = get_gpu_utilization(idx)
            print(f"  GPU {idx}: {info['memory_used_gb']:.2f}/{info['memory_total_gb']:.2f} GB "
                  f"({info['memory_percent']:.1f}%)")
            if "pytorch_max_allocated_gb" in info:
                print(f"         Peak PyTorch allocation: {info['pytorch_max_allocated_gb']:.2f} GB")
    
    # System memory at end
    mem = get_system_memory()
    print(f"\n--- Final System Memory ---")
    print(f"  RAM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB ({mem['percent']:.1f}%)")
    
    print("\n" + "=" * 70)
    
    if errors > 0:
        print("\nFAILED FILES:")
        print("-" * 70)
        for r in results:
            if r["status"] == "error":
                print(f"  ✗ {Path(r['file']).name}")
                print(f"    Error: {r.get('error', 'Unknown error')}")
        print("-" * 70)
    
    # Exit with error code if any failures
    if errors > 0:
        print(f"\n[transcribe] Job completed with {errors} error(s)")
        exit(1)
    else:
        print(f"\n[transcribe] Job completed successfully")


if __name__ == "__main__":
    main()