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
"""

import argparse
import json
import time
import io
import os
from pathlib import Path
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import torchaudio
import numpy as np

from utils import load_config


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


def print_gpu_status(gpus: List[int], prefix: str = "[transcribe]"):
    """Print memory status for all GPUs."""
    for idx in gpus:
        used, free, total = get_gpu_memory_info(idx)
        name = torch.cuda.get_device_name(idx)
        print(f"{prefix} GPU {idx} ({name}): {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")


# ============================================================================
# Audio I/O (No ffmpeg)
# ============================================================================

def load_audio(file_path: Path, target_sr: int = 16000, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using torchaudio (no ffmpeg required for wav/flac/mp3).
    
    For video files (.mp4, .mov), extracts audio track.
    
    Args:
        file_path: Path to audio/video file
        target_sr: Target sample rate
        mono: Convert to mono if True
        
    Returns:
        Tuple of (waveform tensor, sample_rate)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    # For video files, use torchaudio's VideoReader if available, 
    # otherwise fall back to moviepy (no ffmpeg CLI needed)
    if suffix in ('.mp4', '.mov', '.avi', '.mkv'):
        return _load_audio_from_video(file_path, target_sr, mono)
    
    # Standard audio files
    waveform, sr = torchaudio.load(str(file_path))
    
    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    return waveform, sr


def _load_audio_from_video(file_path: Path, target_sr: int, mono: bool) -> Tuple[torch.Tensor, int]:
    """Extract audio from video using moviepy (uses bundled ffmpeg, not system ffmpeg)."""
    try:
        from moviepy.editor import VideoFileClip
        
        clip = VideoFileClip(str(file_path))
        audio = clip.audio
        
        # Get audio as numpy array
        # moviepy's to_soundarray returns (n_samples, n_channels)
        audio_array = audio.to_soundarray(fps=target_sr)
        clip.close()
        
        # Convert to torch tensor (channels, samples)
        waveform = torch.from_numpy(audio_array.T).float()
        
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, target_sr
        
    except ImportError:
        raise ImportError(
            "moviepy is required for video file processing. "
            "Install with: pip install moviepy"
        )


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
    from pyannote.audio import Pipeline
    
    start = time.time()
    pipeline = Pipeline.from_pretrained(
        config.diary_model, 
        use_auth_token=config.hf_token
    )
    
    # Move to device
    device_obj = torch.device(device)
    try:
        pipeline = pipeline.to(device_obj)
    except Exception as e:
        print(f"[transcribe] Warning: Could not move diarization to {device}: {e}")
    
    print(f"[transcribe] Loaded diarization on {device} in {time.time() - start:.1f}s")
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
    
    print(f"[transcribe] Diarization on {device}: {audio_path.name}")
    
    try:
        pipeline = load_diarization_pipeline(config, device)
        
        # Run diarization
        diarization = pipeline(
            str(audio_path), 
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
        print(f"[transcribe] Diarization complete: {audio_path.name} ({elapsed:.1f}s)")
        return rttm_path
        
    except Exception as e:
        print(f"[transcribe] Diarization failed for {audio_path.name}: {e}")
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
    
    start = time.time()
    model = WhisperModel(
        config.whisper_model_size,
        device=device_type,
        device_index=device_index,
        compute_type=compute_type,
    )
    batched = BatchedInferencePipeline(model=model)
    
    print(f"[transcribe] Loaded Whisper '{config.whisper_model_size}' on {device} in {time.time() - start:.1f}s")
    
    if device_type == "cuda":
        used, free, total = get_gpu_memory_info(device_index)
        print(f"[transcribe] GPU {device_index} after model load: {used:.1f}GB used, {free:.1f}GB free")
    
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
    
    print(f"[transcribe] ASR on {device}: {audio_path.name}")
    
    try:
        # Load audio
        waveform, sr = load_audio(audio_path, target_sr=config.audio_sample_rate, mono=True)
        
        # Parse diarization
        segments = parse_rttm(rttm_path)
        
        # Load model
        batched_model = load_whisper_model(config, device)
        
        # Transcribe each segment
        results = []
        temp_chunk = file_subdir / "_temp_chunk.wav"
        
        for seg in segments:
            # Skip if not transcribing both speakers
            if not config.both_speakers and seg["speaker"] != "speaker_1":
                continue
            
            # Skip very short segments
            duration_ms = seg["duration"] * 1000
            if duration_ms < config.min_segment_ms:
                continue
            
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
        print(f"[transcribe] ASR complete: {audio_path.name} ({elapsed:.1f}s, {len(results)} segments)")
        return results_path
        
    except Exception as e:
        print(f"[transcribe] ASR failed for {audio_path.name}: {e}")
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
    print(f"[transcribe] Processing {len(files)} files with {total_workers} workers on {len(gpus_or_cpu)} device(s)")
    
    # Distribute files
    if gpus:
        assignments = distribute_files_to_gpus(files, gpus)
    else:
        assignments = [(f, "cpu") for f in files]
    
    config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
    
    results = []
    
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
        
        for future in as_completed(futures):
            audio_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "file": str(audio_path),
                    "status": "error",
                    "error": str(e)
                })
    
    return results


# ============================================================================
# Main
# ============================================================================

def discover_audio_files(input_dir: Path, temp_audio_dir: Optional[Path] = None) -> List[Path]:
    """Discover audio and video files recursively."""
    extensions = ["*.wav", "*.mp3", "*.flac", "*.mp4", "*.mov", "*.avi"]
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
    
    args = parser.parse_args()
    
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
    print(f"[transcribe] Language: {config.whisper_language}")
    
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
            print_gpu_status(gpus)
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
    
    print(f"[transcribe] Found {len(audio_files)} files to process")
    
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
        print(f"[transcribe] {len(audio_files)} files after filtering existing")
    
    if not audio_files:
        print("[transcribe] Nothing to process")
        return
    
    # Process files
    start = time.time()
    results = process_files_parallel(
        audio_files,
        output_dir,
        config,
        gpus,
        args.stage,
        max_workers_per_gpu=args.workers_per_gpu
    )
    
    # Summary
    elapsed = time.time() - start
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print("\n" + "=" * 60)
    print(f"[transcribe] Complete in {elapsed:.1f}s")
    print(f"  Total files: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print("=" * 60)
    
    if errors > 0:
        print("\nFailed files:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['file']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()


