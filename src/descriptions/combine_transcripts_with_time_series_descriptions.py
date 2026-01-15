"""
Combine time-series descriptions (AU patterns) with transcript summaries (speech content)
using Gemma 7B-it to describe associations.

This script:
1. Loads data_model.yaml containing interview metadata and transcript paths
2. Loads time-series descriptions from generated JSON files (contains "generated_description" key from generate_time_series_descriptions.py)
3. Loads transcript summaries from JSON files (contains "summaries" key from summarize.py)
4. Matches entries by patient_id, interview_type, turn_index, and time window (start_ms, end_ms)
5. Uses Gemma 7B-it to describe associations between AU patterns and speech content
6. Saves combined results to output JSON files
"""

import sys
import os
import torch
import torch.multiprocessing as mp
import json
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# GPU monitoring
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ============================================================================
# GPU MONITORING
# ============================================================================

class GPUMonitor:
    """Monitor GPU memory and utilization."""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.initialized = False
        
        if PYNVML_AVAILABLE:
            try:
                nvmlInit()
                self.handle = nvmlDeviceGetHandleByIndex(gpu_id)
                self.initialized = True
            except Exception as e:
                print(f"⚠️ [GPU {gpu_id}] nvidia-ml-py init failed: {e}")
    
    def get_stats(self) -> Dict:
        """Get GPU memory and utilization stats."""
        stats = {
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'memory_percent': 0,
            'utilization': 0,
            'temperature': 0
        }
        
        if not self.initialized:
            if torch.cuda.is_available():
                try:
                    stats['memory_used_gb'] = torch.cuda.memory_allocated(self.gpu_id) / 1e9
                    stats['memory_total_gb'] = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
                    stats['memory_percent'] = (stats['memory_used_gb'] / stats['memory_total_gb']) * 100
                except Exception:
                    pass
            return stats
        
        try:
            mem_info = nvmlDeviceGetMemoryInfo(self.handle)
            util_info = nvmlDeviceGetUtilizationRates(self.handle)
            temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
            
            stats['memory_used_gb'] = mem_info.used / 1e9
            stats['memory_total_gb'] = mem_info.total / 1e9
            stats['memory_percent'] = (mem_info.used / mem_info.total) * 100
            stats['utilization'] = util_info.gpu
            stats['temperature'] = temp
        except Exception:
            pass
        
        return stats
    
    def format_memory_bar(self, width: int = 25) -> str:
        """Create visual memory bar."""
        stats = self.get_stats()
        filled = int((stats['memory_percent'] / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB ({stats['memory_percent']:.1f}%) | Util: {stats['utilization']}%"
    
    def log_status(self, message: str):
        """Log GPU status with memory information."""
        memory_bar = self.format_memory_bar()
        print(f"🎮 [GPU {self.gpu_id}] {message}")
        print(f"   {memory_bar}")


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    
    # Check CUDA_VISIBLE_DEVICES
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
        return list(range(len(visible.split(','))))
    
    return list(range(torch.cuda.device_count()))


def setup_device():
    """Determine best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_data_model(yaml_path: Path) -> Dict:
    """Load the data_model.yaml file.
    
    Returns:
        Dict mapping video_id -> video metadata (video_path, transcription_path, AUs_speaker_1, AUs_speaker_2, speaker_mapping)
    """
    print(f"\n📂 Loading data model from {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print(f"✅ Loaded {len(data)} videos")
    return data


def _is_nan_like(x) -> bool:
    """Return True if x is None or cannot be interpreted as a finite number (NaN/Inf/string nan)."""
    if x is None:
        return True
    # Handle strings like 'nan'
    if isinstance(x, str):
        try:
            xv = float(x)
        except Exception:
            return True
        return not np.isfinite(xv)
    # Handle numeric types
    try:
        xv = float(x)
    except Exception:
        return True
    return not np.isfinite(xv)


def load_timeseries_descriptions(descriptions_dir: Path) -> Dict[str, List[Dict]]:
    """Load all time-series description JSON files from directory.
    
    The files are output from generate_time_series_descriptions.py and are named:
    - {video_id}_descriptions.json (e.g., downloaded_video_descriptions.json)
    
    Each file contains a list of turns with keys:
    - video_id, turn_index, speaker_id, start_ms, end_ms, duration_ms, text, generated_description
    
    This function needs to map video_id to (patient_id, interview_type) using the data model.
    
    Returns:
        Dict mapping video_id -> list of description entries
    """
    descriptions_by_video = {}
    
    print(f"\n📂 Loading time-series descriptions from {descriptions_dir}...")
    json_files = list(descriptions_dir.glob("*_descriptions.json"))
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract video_id from filename (remove '_descriptions.json')
        video_id = json_file.stem.replace('_descriptions', '')
        descriptions_by_video[video_id] = data
    
    print(f"✅ Loaded {len(json_files)} description files for {len(descriptions_by_video)} videos")
    return descriptions_by_video





def match_entries(
    descriptions: List[Dict],
    summaries: List[Dict],
    tolerance_ms: int = 500
) -> List[Tuple[Dict, Dict]]:
    """Match time-series description and summary entries by turn_index and time window.
    
    Args:
        descriptions: List of time-series description entries from generate_time_series_descriptions.py
        summaries: List of summary entries from summarize.py (extracted from 'summaries' key)
        tolerance_ms: Time window tolerance in milliseconds for matching (default 500ms)
    
    Returns:
        List of (description, summary) tuples for matched entries
    """
    matches = []
    
    # Build lookup by turn_index for summaries
    # Note: summarize.py groups consecutive turns by speaker, so turn_indices is a list
    summaries_indexed = {}
    for summ in summaries:
        turn_indices = summ.get('turn_indices', [])
        if isinstance(turn_indices, list):
            for turn_idx in turn_indices:
                summaries_indexed[turn_idx] = summ
        else:
            # Fallback if turn_indices is a single integer
            summaries_indexed[turn_indices] = summ
    
    for desc in descriptions:
        turn_idx = desc['turn_index']
        start_ms = desc['start_ms']
        end_ms = desc['end_ms']
        
        # Try exact turn index match first
        if turn_idx in summaries_indexed:
            summ = summaries_indexed[turn_idx]
            # Verify time windows overlap (summaries group consecutive speaker turns)
            if (start_ms <= summ['end_ms'] + tolerance_ms and 
                end_ms >= summ['start_ms'] - tolerance_ms):
                matches.append((desc, summ))
                continue
        
        # Fallback: search by time window overlap
        for summ in summaries:
            if (start_ms <= summ['end_ms'] + tolerance_ms and 
                end_ms >= summ['start_ms'] - tolerance_ms):
                matches.append((desc, summ))
                break
    
    return matches


def create_combination_prompt(
    timeseries_description: str, 
    summary: str, 
    speaker_id: str
) -> str:
    """Create prompt for Gemma to combine time-series description and summary."""
    
    prompt = f"""
You are describing the content in a speech turn from a psychotherapy session. 
Your task is to combine what was said with the client and therapist's facial expressions to one short, coherent paragraph.

Data for this turn:

Speech content summary: {summary} (spoken by {speaker_id})

Facial Action Unit (AU) patterns: {timeseries_description}

Instructions:
- Begin by describing the speech content very briefly
- Then briefly note any salient facial Action Units (AUs) that stand out — do not over-analyze every AU, only mention the most relevant ones.
- Do **not** over-analyze or speculate; be very true to what is actually present in the data available. 
- Do not reflect on the emotional bond, synchrony or similar aspects of the interaction.
- Write your description as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.

Description:"""
    
    return prompt


def combine_with_gemma(
    tokenizer,
    model,
    device: str,
    timeseries_description: str,
    summary: str,
    speaker_id: str,
    use_concat: bool = False
) -> str:
    """Use Gemma 7B-it to combine time-series description and summary into coherent text.
    
    Args:
        tokenizer: Gemma tokenizer
        model: Gemma model
        device: Device to run on
        timeseries_description: AU pattern description
        summary: Speech content summary
        speaker_id: Who spoke this turn
        use_concat: If True, simply concatenate without LLM
    
    Returns:
        Combined description text
    """
    
    if use_concat:
        # Simple concatenation bypass
        return f"{summary} {timeseries_description}"
    
    prompt = create_combination_prompt(
        timeseries_description, summary, speaker_id
    )
    
    # Tokenize with proper chat template (Gemma uses specific format)
    messages = [{"role": "user", "content": prompt}]
    
    # Format as chat (some Gemma models use chat template)
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens (not the input)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    combined = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
      
    return combined


def process_video(
    video_id: str,
    video_data: Dict,
    timeseries_descriptions: List[Dict],
    tokenizer,
    model,
    device: str,
    max_turns: Optional[int] = None,
    use_concat: bool = False
) -> List[Dict[str, Any]]:
    """Process all turns for a single video.
    
    Args:
        video_id: Video identifier
        video_data: Video data from data_model.yaml (video_path, transcription_path, AUs_speaker_1, AUs_speaker_2, speaker_mapping)
        timeseries_descriptions: List of time-series description entries from generate_time_series_descriptions.py
        tokenizer: Gemma tokenizer
        model: Gemma model
        device: Device to run on
        max_turns: Maximum number of turns to process (None = all)
        use_concat: If True, use simple concatenation instead of LLM
    """
    
    results = []
    
    # Load summaries from transcript path in data model
    transcript_path = Path(video_data['transcription_path'])
    if not transcript_path.exists():
        print(f"⚠️ Transcript not found: {transcript_path}")
        return results
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Extract summaries (handle both formats)
    if isinstance(transcript_data, dict) and 'summaries' in transcript_data:
        summaries = transcript_data['summaries']
    elif isinstance(transcript_data, list):
        summaries = transcript_data
    else:
        print(f"⚠️ Invalid transcript format for {video_id}")
        return results
    
    # Match time-series descriptions with summaries
    matches = match_entries(timeseries_descriptions, summaries)
    
    if not matches:
        print(f"⚠️ No matches found for {video_id}")
        return results
    
    # Limit turns if requested (for debugging)
    if max_turns is not None and len(matches) > max_turns:
        matches = matches[:max_turns]
        print(f"ℹ️  Limited to first {max_turns} turns for debugging")
    
    print(f"\nProcessing {video_id}: {len(matches)} matched turns")
    
    skipped_empty = 0
    
    for desc, summ in tqdm(matches, desc=f"{video_id}"):
        try:
            # Skip if time-series description is empty or missing
            # Note: generate_time_series_descriptions.py outputs 'generated_description' (singular)
            description_text = desc.get('generated_description', desc.get('generated_descriptions', '')).strip() 
            if not description_text:
                skipped_empty += 1
                continue
            
            # Skip if description is an error message
            if description_text.lower().startswith('error:'):
                skipped_empty += 1
                continue
            
            # Skip if summary is empty
            summary_text = summ.get('summary', summ.get('text', '')).strip()
            if not summary_text:
                skipped_empty += 1
                continue
            
            combined = combine_with_gemma(
                tokenizer,
                model,
                device,
                description_text,
                summary_text,
                desc['speaker_id'],
                use_concat=use_concat
            )
            
            result = {
                "video_id": video_id,
                "turn_index": desc['turn_index'],
                "speaker_id": desc['speaker_id'],
                "start_ms": desc['start_ms'],
                "end_ms": desc['end_ms'],
                "duration_ms": desc['duration_ms'],
                "original_timeseries_description": desc.get('generated_description', desc.get('generated_descriptions', '')),
                "original_summary": summary_text,
                "combined_description": combined
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing turn {desc['turn_index']}: {e}")
            continue
    
    if skipped_empty > 0:
        print(f"⚠️  Skipped {skipped_empty} turn(s) with empty description or summary")
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save combined results to JSON."""
    print(f"\n💾 Saving {len(results)} results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved")


# ============================================================================
# MULTI-GPU WORKER
# ============================================================================

def gpu_worker(
    gpu_id: int,
    video_assignments: List[Tuple[str, Dict]],
    descriptions_by_video: Dict[str, List[Dict]],
    output_dir: Path,
    model_name: str,
    max_turns: Optional[int],
    use_concat: bool,
    skip_existing: bool,
    result_queue: mp.Queue
):
    """Worker function that runs on a specific GPU.
    
    Args:
        gpu_id: GPU index to use
        video_assignments: List of (video_id, video_data) tuples
        descriptions_by_video: Dict mapping video_id to AU descriptions
        output_dir: Output directory
        model_name: Gemma model name
        max_turns: Max turns per video
        use_concat: If True, use simple concatenation
        skip_existing: If True, skip videos with existing outputs
        result_queue: Queue to send results back
    """
    try:
        # Set device for this worker
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        print(f"\n🚀 [GPU {gpu_id}] Worker starting with {len(video_assignments)} videos")
        
        # Initialize GPU monitor
        monitor = GPUMonitor(gpu_id)
        monitor.log_status("Worker starting")
        
        # Load model on this GPU
        if not use_concat:
            print(f"🔧 [GPU {gpu_id}] Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device
            )
            model.eval()
            monitor.log_status("Model loaded")
        else:
            tokenizer = None
            model = None
            print(f"⚡ [GPU {gpu_id}] Using concat mode - no model loaded")
        
        # Process assigned videos
        worker_results = []
        total_turns_processed = 0
        
        for video_idx, (video_id, video_data) in enumerate(video_assignments):
            print(f"\n{'='*60}")
            print(f"[GPU {gpu_id}] Video {video_idx + 1}/{len(video_assignments)}: {video_id}")
            print(f"{'='*60}")
            
            # Check if output already exists
            output_file = output_dir / f"{video_id}_combined.json"
            if skip_existing and output_file.exists():
                print(f"[GPU {gpu_id}] ⏭️  Skipping {video_id} - output already exists")
                continue
            
            # Check if we have descriptions for this video
            if video_id not in descriptions_by_video:
                print(f"[GPU {gpu_id}] ⚠️ No AU descriptions for {video_id}, skipping")
                continue
            
            results = process_video(
                video_id,
                video_data,
                descriptions_by_video[video_id],
                tokenizer,
                model,
                device,
                max_turns=max_turns,
                use_concat=use_concat
            )
            
            # Save immediately after each video
            if results:
                save_results(results, output_file)
                worker_results.append({
                    'video_id': video_id,
                    'num_turns': len(results),
                    'output_path': str(output_file)
                })
                total_turns_processed += len(results)
            
            # Log GPU stats periodically
            if (video_idx + 1) % 3 == 0 or video_idx == len(video_assignments) - 1:
                monitor.log_status(f"Progress: {video_idx + 1}/{len(video_assignments)} videos")
        
        # Clean up
        if model is not None:
            del model
            del tokenizer
        torch.cuda.empty_cache()
        
        # Send results back
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


def distribute_videos(
    video_ids: List[str],
    data_model: Dict,
    num_gpus: int
) -> Dict[int, List[Tuple[str, Dict]]]:
    """Distribute videos across GPUs using round-robin assignment.
    
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
        description="Combine time-series descriptions with transcript summaries using Gemma 7B-it"
    )
    parser.add_argument(
        "--data_model",
        type=Path,
        required=True,
        help="Path to data_model.yaml (output from map_speaker_to_aus.py)"
    )
    parser.add_argument(
        "--descriptions_dir",
        type=Path,
        required=True,
        help="Directory containing time-series description JSON files (output from generate_time_series_descriptions.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for combined results"
    )
    parser.add_argument( 
        "--model_name",
        type=str,
        default="google/gemma-7b-it",
        help="Gemma model to use (default: google/gemma-7b-it)"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (None = all)"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Maximum number of speech turns to process per video (None = all, useful for debugging)"
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Use simple string concatenation instead of LLM (bypasses Gemma model)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have output JSON files"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU indices to use (e.g., '0,1,2,3' for 4 A100s). Default: 0,1,2,3"
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Force single-GPU mode (no multiprocessing)"
    )
    
    args = parser.parse_args()
    
    # Determine GPUs to use
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = get_available_gpus()
    
    if not gpu_ids and not args.concat:
        print("❌ No GPUs available! Use --concat for CPU-only mode")
        sys.exit(1)
    
    num_gpus = len(gpu_ids) if not args.single_gpu else min(1, len(gpu_ids))
    
    print("🚀 Starting time-series description + summary combination" + (" (concat mode)" if args.concat else " with Gemma 7B-it"))
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data model: {args.data_model}")
    print(f"  Time-series descriptions dir: {args.descriptions_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Mode: {'String concatenation' if args.concat else f'LLM ({args.model_name})'}")  
    print(f"  GPUs: {gpu_ids if gpu_ids else 'CPU only'}")
    print(f"  Multi-GPU mode: {'No (single-GPU)' if args.single_gpu or num_gpus <= 1 else f'Yes ({num_gpus} GPUs)'}")
    print(f"  Skip existing: {args.skip_existing}")
    if args.max_turns:
        print(f"  Max turns per video: {args.max_turns} (debugging mode)")
    print()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_model = load_data_model(args.data_model)
    descriptions_by_video = load_timeseries_descriptions(args.descriptions_dir)
    
    video_ids = list(data_model.keys())
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]
    
    print(f"\n📊 Processing {len(video_ids)} videos")
    
    # =========================================================================
    # SINGLE-GPU MODE (or concat mode)
    # =========================================================================
    if args.single_gpu or num_gpus <= 1 or args.concat:
        device = f"cuda:{gpu_ids[0]}" if gpu_ids and not args.concat else "cpu"
        print(f"\n🔧 Using device: {device}")
        
        if gpu_ids and not args.concat:
            torch.cuda.set_device(gpu_ids[0])
        
        # Load Gemma model (skip if using concatenation)
        if args.concat:
            print("\n⚡ Using simple concatenation mode - skipping model load")
            tokenizer = None
            model = None
        else:
            print(f"\n🔧 Loading {args.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            model.eval()
            
            print(f"✅ Model loaded on {device}")
            if device != "cpu":
                print(f"   GPU Memory allocated: {torch.cuda.memory_allocated(gpu_ids[0]) / 1024**3:.2f} GB")
        
        # Process all videos
        all_results = []
        processed_count = 0
        skipped_count = 0
        
        for video_idx, video_id in enumerate(video_ids):
            print(f"\n{'='*80}")
            print(f"Video {video_idx + 1}/{len(video_ids)}: {video_id}")
            print(f"{'='*80}")
            
            # Check if output already exists
            output_file = args.output_dir / f"{video_id}_combined.json"
            if args.skip_existing and output_file.exists():
                print(f"⏭️  Skipping {video_id} - output already exists")
                skipped_count += 1
                continue
            
            # Check if we have time-series descriptions for this video
            if video_id not in descriptions_by_video:
                print(f"⚠️ No time-series descriptions found for {video_id}, skipping")
                continue
            
            video_data = data_model[video_id]
            
            results = process_video(
                video_id,
                video_data,
                descriptions_by_video[video_id],
                tokenizer,
                model,
                device,
                max_turns=args.max_turns,
                use_concat=args.concat
            )
            
            if results:
                all_results.extend(results)
                processed_count += 1
                
                # Save per video
                save_results(results, output_file)
        
        print(f"\n✅ Complete! Combined {len(all_results)} turns across {processed_count} videos")
        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} videos (already processed)")
        print(f"📁 Results saved to: {args.output_dir}")
    
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
                    descriptions_by_video,
                    args.output_dir,
                    args.model_name,
                    args.max_turns,
                    args.concat,
                    args.skip_existing,
                    result_queue
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
        print(f"  📁 Results saved to: {args.output_dir}")
if __name__ == "__main__":
    main()