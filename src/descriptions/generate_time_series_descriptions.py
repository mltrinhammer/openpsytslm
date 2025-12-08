"""
This script generates time-series descriptions of the AU.
In this step, we give a Gemma-3 a heatmap of action unit time series, and simply asks it to describe it.
Its akin to what the opentslm authors did with gpt-4o.

Multi-GPU support for H100 clusters with GPU memory/utilization logging.
"""


import sys
import os
import time
import threading
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import random
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from queue import Queue
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline
import yaml 

# GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ============================================================================
# GPU MONITORING UTILITIES
# ============================================================================

class GPUMonitor:
    """Monitor GPU memory and utilization with nice logging."""
    
    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.initialized = False
        self._lock = threading.Lock()
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handles = {gpu_id: pynvml.nvmlDeviceGetHandleByIndex(gpu_id) 
                               for gpu_id in gpu_ids}
                self.initialized = True
            except Exception as e:
                print(f"⚠️ pynvml initialization failed: {e}")
        else:
            print("⚠️ pynvml not available. Install with: pip install pynvml")
    
    def get_gpu_stats(self, gpu_id: int) -> Dict[str, Any]:
        """Get memory and utilization stats for a GPU."""
        stats = {
            'gpu_id': gpu_id,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'memory_percent': 0,
            'utilization': 0,
            'temperature': 0
        }
        
        if not self.initialized or gpu_id not in self.handles:
            # Fallback to torch.cuda
            if torch.cuda.is_available():
                try:
                    stats['memory_used_gb'] = torch.cuda.memory_allocated(gpu_id) / 1e9
                    stats['memory_total_gb'] = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                    stats['memory_percent'] = (stats['memory_used_gb'] / stats['memory_total_gb']) * 100
                except Exception:
                    pass
            return stats
        
        try:
            with self._lock:
                handle = self.handles[gpu_id]
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                stats['memory_used_gb'] = mem_info.used / 1e9
                stats['memory_total_gb'] = mem_info.total / 1e9
                stats['memory_percent'] = (mem_info.used / mem_info.total) * 100
                stats['utilization'] = util_info.gpu
                stats['temperature'] = temp
        except Exception as e:
            pass
        
        return stats
    
    def format_memory_bar(self, gpu_id: int, width: int = 30) -> str:
        """Create a visual memory bar for a GPU."""
        stats = self.get_gpu_stats(gpu_id)
        filled = int((stats['memory_percent'] / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"GPU {gpu_id}: [{bar}] {stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%) | Util: {stats['utilization']}% | Temp: {stats['temperature']}°C"
    
    def log_all_gpus(self, prefix: str = ""):
        """Log stats for all GPUs."""
        print(f"\n{'='*80}")
        if prefix:
            print(f"📊 {prefix}")
        for gpu_id in self.gpu_ids:
            print(f"  {self.format_memory_bar(gpu_id)}")
        print(f"{'='*80}\n")
    
    def shutdown(self):
        """Clean up pynvml."""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    
    # Check CUDA_VISIBLE_DEVICES
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
        return list(range(len(visible.split(','))))
    
    return list(range(torch.cuda.device_count()))


def estimate_workers_per_gpu(model_size_gb: float = 54.0, gpu_memory_gb: float = 80.0) -> int:
    """
    Estimate how many workers can run on each GPU.
    
    27B params at bfloat16 = ~54GB
    H100 has 80GB, leaving ~26GB headroom
    
    Since the model is shared within a process, we can't easily run multiple
    workers on the same GPU with separate model instances. Instead, we'll
    process videos sequentially on each GPU but distribute videos across GPUs.
    """
    available = gpu_memory_gb - model_size_gb
    # We need headroom for inference (activations, KV cache, etc.)
    # With 26GB headroom on H100, we can comfortably do inference
    # but multiple parallel inferences would require more sophisticated memory management
    return 1  # Safe default - one worker per GPU


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file."""
    print(f"Loading data model from {yaml_path}...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data)} videos")
    return data


def load_speech_turns(json_path: Path) -> List[Dict]:
    """Load speech turns from transcript JSON."""
    with open(json_path, 'r') as f:
        turns = json.load(f)
    return turns


def extract_au_window(csv_path: Path, start_ms: float, end_ms: float, au_columns: List[str]) -> pd.DataFrame:
    """Extract AU data from OpenFace CSV for a specific time window.
    
    Args:
        csv_path: Path to OpenFace CSV
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        au_columns: List of AU column names to extract
    
    Returns:
        DataFrame with timestamp and AU columns for the specified window
    """
    # Read CSV with whitespace handling
    df = pd.read_csv(csv_path, skipinitialspace=True)
    
    # Convert timestamp from seconds to milliseconds
    df['timestamp_ms'] = df['timestamp'] * 1000
    
    # Filter to time window
    mask = (df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)
    window_df = df.loc[mask, ['timestamp_ms'] + au_columns].copy()
    
    return window_df


def bin_time_series(data: pd.DataFrame, au_name: str, num_bins: int = 8) -> np.ndarray:
    """Bin time series data into equal temporal bins and compute mean activation per bin.
    
    Args:
        data: DataFrame with timestamp_ms and AU columns
        au_name: Name of the AU column to bin
        num_bins: Number of temporal bins
    
    Returns:
        Array of mean activations per bin (length = num_bins)
    """
    if len(data) == 0:
        return np.zeros(num_bins)
    
    # Create bin indices for each row
    bin_indices = np.linspace(0, len(data), num_bins + 1, dtype=int)
    
    # Compute mean activation for each bin
    binned_values = []
    for i in range(num_bins):
        start_idx = bin_indices[i]
        end_idx = bin_indices[i + 1]
        if end_idx > start_idx:
            bin_mean = data[au_name].iloc[start_idx:end_idx].mean()
            binned_values.append(bin_mean)
        else:
            binned_values.append(0.0)
    
    return np.array(binned_values)


def generate_plot_for_turn(
    speaker1_csv: Path,
    speaker2_csv: Path,
    turn: Dict,
    au_names: List[str],
    output_path: Path,
    num_bins: int = 8
) -> bool:
    """Generate heatmap visualization with binned AU activations for speaker 1 and speaker 2.
    
    Args:
        speaker1_csv: Path to speaker 1 OpenFace CSV
        speaker2_csv: Path to speaker 2 OpenFace CSV
        turn: Speech turn dict with start_ms, end_ms, speaker_id
        au_names: List of AU names to plot (can be 4 or 17)
        output_path: Where to save the plot
        num_bins: Number of temporal bins (default 8)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        start_ms = turn['start_ms']
        end_ms = turn['end_ms']
        speaker_id = turn['speaker_id']
        turn_index = turn['turn_index']
        
        # Extract AU data for both speakers
        speaker1_data = extract_au_window(speaker1_csv, start_ms, end_ms, au_names)
        speaker2_data = extract_au_window(speaker2_csv, start_ms, end_ms, au_names)
        
        # Check if we have data
        if speaker1_data.empty or speaker2_data.empty:
            print(f"⚠️ No data found for turn {turn_index} ({start_ms}-{end_ms}ms)")
            return False
        
        # Create binned heatmaps for each AU
        speaker1_heatmap = np.zeros((len(au_names), num_bins))
        speaker2_heatmap = np.zeros((len(au_names), num_bins))
        
        for i, au_name in enumerate(au_names):
            speaker1_heatmap[i, :] = bin_time_series(speaker1_data, au_name, num_bins)
            speaker2_heatmap[i, :] = bin_time_series(speaker2_data, au_name, num_bins)
        
        # Create figure with two side-by-side heatmaps
        # Dynamically adjust figure height based on number of AUs
        fig_height = max(6, len(au_names) * 0.5)  # At least 6 inches, scale with AU count
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))
        
        # Find global min/max for consistent color scaling
        vmin = min(speaker1_heatmap.min(), speaker2_heatmap.min())
        vmax = max(speaker1_heatmap.max(), speaker2_heatmap.max())
        
        # Speaker 1 heatmap (left)
        im1 = ax1.imshow(speaker1_heatmap, aspect='auto', cmap='Blues', 
                         interpolation='nearest', vmin=vmin, vmax=vmax)
        ax1.set_title('SPEAKER 1 AU Activation', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Action Unit', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Progression (Start → End)', fontsize=12, fontweight='bold')
        ax1.set_yticks(range(len(au_names)))
        ax1.set_yticklabels(au_names, fontsize=11)
        ax1.set_xticks(range(num_bins))
        ax1.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Late-Mid', 'Late', 'Very Late', 'End'][:num_bins], 
                            fontsize=9, rotation=45, ha='right')
        
        # Add colorbar for speaker 1
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Activation Level', fontsize=11, fontweight='bold')
        
        # Add value annotations on speaker 1 heatmap (smaller font for many AUs)
        font_size = 8 if len(au_names) <= 4 else 6
        for i in range(len(au_names)):
            for j in range(num_bins):
                text = ax1.text(j, i, f'{speaker1_heatmap[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=font_size)
        
        # Speaker 2 heatmap (right)
        im2 = ax2.imshow(speaker2_heatmap, aspect='auto', cmap='Oranges', 
                         interpolation='nearest', vmin=vmin, vmax=vmax)
        ax2.set_title('SPEAKER 2 AU Activation', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Action Unit', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Progression (Start → End)', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(au_names)))
        ax2.set_yticklabels(au_names, fontsize=11)
        ax2.set_xticks(range(num_bins))
        ax2.set_xticklabels(['Start', 'Early', 'Early-Mid', 'Mid', 'Mid-Late', 'Late', 'Very Late', 'End'][:num_bins], 
                            fontsize=9, rotation=45, ha='right')
        
        # Add colorbar for speaker 2
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Activation Level', fontsize=11, fontweight='bold')
        
        # Add value annotations on speaker 2 heatmap (smaller font for many AUs)
        for i in range(len(au_names)):
            for j in range(num_bins):
                text = ax2.text(j, i, f'{speaker2_heatmap[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=font_size)
        
        # Overall title
        plt.suptitle(f"Turn {turn_index}: {speaker_id.capitalize()} speaking ({start_ms:.0f}-{end_ms:.0f}ms)\n" + 
                     f"Heatmap shows mean AU activation across {num_bins} temporal phases", 
                     fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating plot for turn {turn.get('turn_index', '?')}: {e}")
        return False



def generate_description_with_pipeline(pipe, image_path: Path, turn: Dict, au_names: List[str]) -> str:
    """Generate time-series description using the Gemma-3 pipeline."""
    
    pre_prompt = """Describe these AU heatmaps (left=speaker 1 blue, right=speaker 2 orange). 
Each row is one AU across 8 time bins. Write one compact sentence per AU comparing patterns.
ONLY consider the AUs which shows either 1) high variability within speaker 1 or speaker 2 or 2) strong difference between speaker 1 and speaker 2
Consider a maximum of 4 action units. 
Format: "AU##: speaker 1 [pattern], speaker 2 [pattern], [key difference]."
No markdown, bullets, or headers. ONLY output your description."""
    
    au_list = ", ".join(au_names)
    context = f"""Turn {turn['turn_index']} ({turn['speaker_id']}), {turn['start_ms']:.0f}-{turn['end_ms']:.0f}ms
AUs: {au_list}
Description:"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": f"{pre_prompt}\n\n{context}"}
            ]
        }
    ]
    
    try:
        # Fixed token limit: model instructed to focus only on most volatile AUs
        output = pipe(
            text=messages,
            max_new_tokens=80,
            do_sample=False
        )
        description = output[0]["generated_text"][-1]["content"].strip()
        
        # Optimized: Minimal post-processing - remove only essential unwanted elements
        # Remove common prefixes
        if description.lower().startswith(("here's", "here is", "the patterns")):
            if ':' in description[:50]:
                description = description.split(':', 1)[1].lstrip()
        
        # Remove markdown and normalize whitespace
        import re
        description = description.replace('**', '').replace('*', '')
        description = description.replace('\n', ' ')
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description
    except Exception as e:
        print(f"❌ Error generating description: {e}")
        return f"Error: {str(e)}"


def convert_transcript_to_turns(transcript_data: List[Dict]) -> List[Dict]:
    """Convert transcript JSON to turn format with turn_index and times in ms.
    
    Args:
        transcript_data: List of transcript segments from transcribe.py
                         Format: [{"text": str, "start": float, "end": float, "speaker": str}, ...]
    
    Returns:
        List of turns with turn_index, speaker_id, start_ms, end_ms
    """
    turns = []
    for idx, segment in enumerate(transcript_data):
        turn = {
            "turn_index": idx,
            "speaker_id": segment["speaker"],  # speaker_1 or speaker_2
            "start_ms": segment["start"] * 1000,  # Convert seconds to ms
            "end_ms": segment["end"] * 1000,
            "text": segment["text"]
        }
        turns.append(turn)
    return turns


def process_video(
    video_id: str,
    video_data: Dict,
    pipe,
    output_dir: Path,
    au_names: List[str],
    max_turns: int = None
) -> List[Dict[str, Any]]:
    """Process a single video for description generation.
    
    Args:
        video_id: Video identifier (stem)
        video_data: Video data dict from data_model.yaml with keys:
                    - video_path, transcription_path, AUs_speaker_1, AUs_speaker_2, speaker_mapping
        pipe: Gemma-3 pipeline
        output_dir: Directory to save plots and results
        au_names: List of AU names to analyze
        max_turns: Maximum number of turns to process (None = all)
    
    Returns:
        List of results dicts
    """
    speaker1_csv = Path(video_data['AUs_speaker_1'])
    speaker2_csv = Path(video_data['AUs_speaker_2'])
    transcript_json = Path(video_data['transcription_path'])
    
    # Validate paths
    if not speaker1_csv.exists():
        print(f"❌ Speaker 1 AU CSV not found: {speaker1_csv}")
        return []
    if not speaker2_csv.exists():
        print(f"❌ Speaker 2 AU CSV not found: {speaker2_csv}")
        return []
    if not transcript_json.exists():
        print(f"❌ Transcript JSON not found: {transcript_json}")
        return []
    
    # Load and convert transcript to turns
    transcript_data = load_speech_turns(transcript_json)
    turns = convert_transcript_to_turns(transcript_data)
    
    # Limit turns if requested
    if max_turns:
        turns = turns[:max_turns]
    
    results = []
    
    # Create .temp directory for plots
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {video_id}: {len(turns)} turns")
    
    for turn in tqdm(turns, desc=f"{video_id}"):
        turn_index = turn['turn_index']
        
        # Generate plot in .temp directory
        plot_path = temp_dir / f"{video_id}_turn{turn_index:03d}.jpg"
        success = generate_plot_for_turn(speaker1_csv, speaker2_csv, turn, au_names, plot_path)
        
        if not success:
            continue
        
        # Generate description
        description = generate_description_with_pipeline(pipe, plot_path, turn, au_names)
        
        # Collect result
        result = {
            "video_id": video_id,
            "turn_index": turn_index,
            "speaker_id": turn['speaker_id'],
            "start_ms": turn['start_ms'],
            "end_ms": turn['end_ms'],
            "duration_ms": turn['end_ms'] - turn['start_ms'],
            "text": turn['text'],
            "generated_description": description,  
            "plot_path": str(plot_path),
            "speaker_1_au_path": str(speaker1_csv),
            "speaker_2_au_path": str(speaker2_csv)
        }
        results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save the generated time-series descriptions to JSON."""
    print(f"\n💾 Saving {len(results)} results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved")
    
    if results:
        avg_duration = np.mean([r['duration_ms'] for r in results])
        avg_description_len = np.mean([len(r['generated_description']) for r in results])
        print(f"\n📊 Summary:")
        print(f"  Total turns processed: {len(results)}")
        print(f"  Average turn duration: {avg_duration:.0f}ms")
        print(f"  Average description length: {avg_description_len:.0f} characters")


# ============================================================================
# MULTI-GPU WORKER
# ============================================================================

def gpu_worker(
    gpu_id: int,
    video_assignments: List[Tuple[str, Dict]],
    output_dir: Path,
    au_columns: List[str],
    max_turns_per_video: Optional[int],
    result_queue: mp.Queue,
    log_interval: int = 10
):
    """
    Worker function that runs on a specific GPU.
    
    Args:
        gpu_id: GPU index to use
        video_assignments: List of (video_id, video_data) tuples assigned to this GPU
        output_dir: Output directory for results
        au_columns: AU columns to analyze
        max_turns_per_video: Max turns per video (None = all)
        result_queue: Queue to send results back to main process
        log_interval: Log GPU stats every N turns
    """
    try:
        # Set device for this worker
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        print(f"\n🚀 [GPU {gpu_id}] Worker starting with {len(video_assignments)} videos")
        
        # Initialize GPU monitor for this worker
        monitor = GPUMonitor([gpu_id])
        
        # Log GPU stats before model loading
        print(f"📊 [GPU {gpu_id}] Memory before model load:")
        print(f"  {monitor.format_memory_bar(gpu_id)}")
        
        # Load model on this GPU
        start_time = time.time()
        print(f"🔧 [GPU {gpu_id}] Loading Gemma-3-27b-it model...")
        
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-27b-it",
            device=device,
            torch_dtype=torch.bfloat16
        )
        
        load_time = time.time() - start_time
        print(f"✅ [GPU {gpu_id}] Model loaded in {load_time:.1f}s")
        print(f"📊 [GPU {gpu_id}] Memory after model load:")
        print(f"  {monitor.format_memory_bar(gpu_id)}")
        
        # Process assigned videos
        worker_results = []
        total_turns_processed = 0
        
        for video_idx, (video_id, video_data) in enumerate(video_assignments):
            print(f"\n{'='*60}")
            print(f"[GPU {gpu_id}] Video {video_idx + 1}/{len(video_assignments)}: {video_id}")
            print(f"{'='*60}")
            
            results = process_video(
                video_id,
                video_data,
                pipe,
                output_dir,
                au_columns,
                max_turns=max_turns_per_video
            )
            
            # Save immediately after each video
            if results:
                output_json = output_dir / f"{video_id}_descriptions.json"
                save_results(results, output_json)
                worker_results.append({
                    'video_id': video_id,
                    'num_turns': len(results),
                    'output_path': str(output_json)
                })
                total_turns_processed += len(results)
            
            # Log GPU stats periodically
            if (video_idx + 1) % log_interval == 0 or video_idx == len(video_assignments) - 1:
                print(f"\n📊 [GPU {gpu_id}] Progress checkpoint ({video_idx + 1}/{len(video_assignments)} videos):")
                print(f"  {monitor.format_memory_bar(gpu_id)}")
                print(f"  Total turns processed: {total_turns_processed}")
        
        # Clean up
        del pipe
        torch.cuda.empty_cache()
        monitor.shutdown()
        
        # Send results back to main process
        result_queue.put({
            'gpu_id': gpu_id,
            'status': 'success',
            'results': worker_results,
            'total_turns': total_turns_processed
        })
        
        print(f"\n✅ [GPU {gpu_id}] Worker completed: {total_turns_processed} turns across {len(worker_results)} videos")
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"\n❌ [GPU {gpu_id}] Worker failed: {e}")
        print(error_msg)
        result_queue.put({
            'gpu_id': gpu_id,
            'status': 'error',
            'error': str(e),
            'traceback': error_msg
        })


def distribute_videos(video_ids: List[str], data_model: Dict, num_gpus: int) -> Dict[int, List[Tuple[str, Dict]]]:
    """
    Distribute videos across GPUs using round-robin assignment.
    
    Returns:
        Dict mapping gpu_id -> list of (video_id, video_data) tuples
    """
    assignments = {gpu_id: [] for gpu_id in range(num_gpus)}
    
    for idx, video_id in enumerate(video_ids):
        gpu_id = idx % num_gpus
        assignments[gpu_id].append((video_id, data_model[video_id]))
    
    # Log distribution
    print("\n📋 Video distribution across GPUs:")
    for gpu_id, videos in assignments.items():
        print(f"  GPU {gpu_id}: {len(videos)} videos")
    
    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Generate AU time-series descriptions from data_model.yaml using Gemma-3 multimodal pipeline"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        required=True,
        help="Path to data_model.yaml (output from map_speaker_to_aus.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for plots and results"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (None = all)"
    )
    parser.add_argument(
        "--max_turns_per_video",
        type=int,
        default=None,
        help="Maximum turns per video (None = all)"
    )
    parser.add_argument(
        "--au_columns",
        nargs="+",
        default=['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r'],
        help="AU columns to analyze (default: 4 key AUs)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices to use (e.g., '0,1'). Default: all available"
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Force single-GPU mode (no multiprocessing)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="Log GPU stats every N videos (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting AU time-series description generation with Gemma-3 multimodal pipeline")
    print("=" * 80)
    
    # Determine GPUs to use
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = get_available_gpus()
    
    if not gpu_ids:
        print("❌ No GPUs available! Falling back to CPU (will be very slow)")
        gpu_ids = []
    
    num_gpus = len(gpu_ids) if not args.single_gpu else 1
    
    print(f"\n📊 Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  AU columns: {args.au_columns}")
    print(f"  GPUs: {gpu_ids if gpu_ids else 'CPU only'}")
    print(f"  Mode: {'Single-GPU' if args.single_gpu or num_gpus <= 1 else f'Multi-GPU ({num_gpus} GPUs)'}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data model
    data_model = load_data_model(args.data_model)
    video_ids = list(data_model.keys())
    
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]
    
    print(f"\n📹 Videos to process: {len(video_ids)}")
    
    # Initialize GPU monitor
    monitor = GPUMonitor(gpu_ids) if gpu_ids else None
    
    if monitor:
        monitor.log_all_gpus("Initial GPU Status")
    
    # =========================================================================
    # SINGLE-GPU MODE (or CPU fallback)
    # =========================================================================
    if args.single_gpu or num_gpus <= 1:
        device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
        print(f"\n🔧 Using device: {device}")
        
        if gpu_ids:
            torch.cuda.set_device(gpu_ids[0])
        
        # Load model
        start_time = time.time()
        print(f"\n🔧 Loading Gemma-3-27b-it model...")
        
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-27b-it",
            device=device,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        
        if monitor:
            monitor.log_all_gpus("After Model Load")
        
        # Process all videos
        all_results = []
        saved_files = []
        
        for video_idx, video_id in enumerate(video_ids):
            video_data = data_model[video_id]
            
            print(f"\n{'='*80}")
            print(f"Video {video_idx + 1}/{len(video_ids)}: {video_id}")
            print(f"{'='*80}")
            
            results = process_video(
                video_id,
                video_data,
                pipe,
                args.output_dir,
                args.au_columns,
                max_turns=args.max_turns_per_video
            )
            all_results.extend(results)
            
            # Save immediately after each video
            if results:
                output_json = args.output_dir / f"{video_id}_descriptions.json"
                save_results(results, output_json)
                saved_files.append(str(output_json))
            
            # Log GPU stats periodically
            if monitor and ((video_idx + 1) % args.log_interval == 0 or video_idx == len(video_ids) - 1):
                monitor.log_all_gpus(f"Progress: {video_idx + 1}/{len(video_ids)} videos")
        
        print(f"\n✅ Complete! Generated descriptions for {len(all_results)} speech turns across {len(saved_files)} videos")
    
    # =========================================================================
    # MULTI-GPU MODE
    # =========================================================================
    else:
        print(f"\n🚀 Launching {num_gpus} GPU workers...")
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Distribute videos across GPUs
        video_assignments = distribute_videos(video_ids, data_model, num_gpus)
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Launch worker processes
        processes = []
        for gpu_id in gpu_ids[:num_gpus]:
            p = mp.Process(
                target=gpu_worker,
                args=(
                    gpu_id,
                    video_assignments[gpu_id],
                    args.output_dir,
                    args.au_columns,
                    args.max_turns_per_video,
                    result_queue,
                    args.log_interval
                )
            )
            p.start()
            processes.append(p)
            print(f"  Started worker on GPU {gpu_id} (PID: {p.pid})")
        
        # Wait for all workers to complete
        print(f"\n⏳ Waiting for {len(processes)} workers to complete...")
        
        # Collect results from queue
        all_worker_results = []
        for _ in range(len(processes)):
            result = result_queue.get()
            all_worker_results.append(result)
            if result['status'] == 'success':
                print(f"  ✅ GPU {result['gpu_id']} completed: {result['total_turns']} turns")
            else:
                print(f"  ❌ GPU {result['gpu_id']} failed: {result['error']}")
        
        # Wait for processes to terminate
        for p in processes:
            p.join()
        
        # Summary
        total_turns = sum(r.get('total_turns', 0) for r in all_worker_results if r['status'] == 'success')
        total_videos = sum(len(r.get('results', [])) for r in all_worker_results if r['status'] == 'success')
        failed_gpus = [r['gpu_id'] for r in all_worker_results if r['status'] == 'error']
        
        print(f"\n{'='*80}")
        print("🎉 MULTI-GPU PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"  Total videos processed: {total_videos}")
        print(f"  Total turns processed: {total_turns}")
        print(f"  GPUs used: {num_gpus}")
        if failed_gpus:
            print(f"  ⚠️ Failed GPUs: {failed_gpus}")
        
        if monitor:
            monitor.log_all_gpus("Final GPU Status")
    
    if monitor:
        monitor.shutdown()


if __name__ == "__main__":
    main()
