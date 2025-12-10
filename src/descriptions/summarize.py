import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.multiprocessing as mp

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


def group_text_by_speaker_turns(turns: List[dict]) -> List[Dict]:
    """Group consecutive turns by the same speaker into speaker turns.
    
    Returns:
        List of dicts with keys: speaker_id, text, start_ms, end_ms, turn_indices
    """
    if not turns:
        return []
    
    speaker_turns = []
    current_speaker = None
    current_texts = []
    current_start_ms = None
    current_end_ms = None
    current_turn_indices = []
    
    for idx, turn in enumerate(turns):
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        
        # Handle both speaker_1/speaker_2 and speaker/speaker_id formats
        speaker_id = turn.get("speaker", turn.get("speaker_id", "unknown"))
        speaker_id = str(speaker_id).strip().lower()
        
        # Convert seconds to ms if needed
        start = turn.get("start", 0)
        end = turn.get("end", start)
        
        # Detect if values are in seconds (< 10000) or ms
        if isinstance(start, (int, float)) and start < 10000:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
        else:
            start_ms = int(start)
            end_ms = int(end)
        
        # If speaker changes or first turn, start a new speaker turn
        if speaker_id != current_speaker:
            # Save previous speaker turn if exists
            if current_speaker is not None and current_texts:
                speaker_turns.append({
                    "speaker_id": current_speaker,
                    "text": " ".join(current_texts),
                    "start_ms": current_start_ms,
                    "end_ms": current_end_ms,
                    "turn_indices": current_turn_indices
                })
            
            # Start new speaker turn
            current_speaker = speaker_id
            current_texts = [text]
            current_start_ms = start_ms
            current_end_ms = end_ms
            current_turn_indices = [idx]
        else:
            # Same speaker, concatenate text
            current_texts.append(text)
            current_end_ms = end_ms  # Update end time to latest
            current_turn_indices.append(idx)
    
    # Don't forget the last speaker turn
    if current_speaker is not None and current_texts:
        speaker_turns.append({
            "speaker_id": current_speaker,
            "text": " ".join(current_texts),
            "start_ms": current_start_ms,
            "end_ms": current_end_ms,
            "turn_indices": current_turn_indices
        })
    
    return speaker_turns


def create_summary_prompt(speaker_id: str, start_ms: int, end_ms: int, combined_text: str) -> str:
    """Create prompt for summarizing speaker turn."""
    start_seconds = max(start_ms, 0) // 1000
    end_seconds = max(end_ms, 0) // 1000
    start_minute, start_second = divmod(start_seconds, 60)
    end_minute, end_second = divmod(end_seconds, 60)
    time_range = f"Time {start_minute:02d}:{start_second:02d}–{end_minute:02d}:{end_second:02d}"
    
    # Map speaker IDs to readable labels
    if speaker_id in ("speaker_1", "speaker1"):
        speaker_label = "Speaker 1"
    elif speaker_id in ("speaker_2", "speaker2"):
        speaker_label = "Speaker 2"
    else:
        speaker_label = speaker_id.replace("_", " ").title()
    
    return (
        f"You are a concise conversation summarizer. "
        f"Provide a short, anonymous summary (1-2 sentences) of what {speaker_label} said.\n"
        f"{speaker_label}'s speech excerpt:\n"
        f"{combined_text}\n"
        f"Summary:"
    )



def gpu_worker(gpu_id: int, video_paths: List[Path], model_name: str, skip_existing: bool):
    """Worker process for GPU inference."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    torch.cuda.set_device(gpu_id)
    
    monitor = GPUMonitor(gpu_id)
    monitor.log_status("Worker starting")
    
    # Load model
    print(f"[GPU {gpu_id}] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}"
    )
    model.eval()
    
    monitor.log_status("Model loaded")
    
    # Process videos
    for video_idx, video_path in enumerate(video_paths):
        summary_path = video_path.with_suffix(".summary.json")
        
        # Skip if exists and flag is set
        if skip_existing and summary_path.exists():
            print(f"[GPU {gpu_id}] Skipping {video_path.name} (summary exists)")
            continue
        
        print(f"[GPU {gpu_id}] Processing {video_path.name} ({video_idx + 1}/{len(video_paths)})")
        
        try:
            # Load transcription
            transcript_path = video_path.with_suffix(".json")
            if not transcript_path.exists():
                print(f"[GPU {gpu_id}] No transcript found for {video_path.name}, skipping")
                continue
            
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            
            turns = transcript_data.get("turns", [])
            if not turns:
                print(f"[GPU {gpu_id}] No turns in transcript for {video_path.name}, skipping")
                continue
            
            # Group by speaker turns
            speaker_turns = group_text_by_speaker_turns(turns)
            if not speaker_turns:
                print(f"[GPU {gpu_id}] No speaker turns for {video_path.name}, skipping")
                continue
            
            # Generate summaries
            summaries = []
            for st in speaker_turns:
                prompt = create_summary_prompt(
                    st["speaker_id"],
                    st["start_ms"],
                    st["end_ms"],
                    st["text"]
                )
                
                inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{gpu_id}")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the summary part after "Summary:"
                if "Summary:" in generated_text:
                    summary_text = generated_text.split("Summary:")[-1].strip()
                else:
                    summary_text = generated_text.strip()
                
                summaries.append({
                    "speaker_id": st["speaker_id"],
                    "start_ms": st["start_ms"],
                    "end_ms": st["end_ms"],
                    "turn_indices": st["turn_indices"],
                    "original_text": st["text"],
                    "summary": summary_text
                })
            
            # Save summaries
            output_data = {
                "video_path": str(video_path),
                "summaries": summaries
            }
            
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"[GPU {gpu_id}] Saved {len(summaries)} summaries to {summary_path.name}")
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {video_path.name}: {e}")
            continue
        
        # Log status every 5 videos
        if (video_idx + 1) % 5 == 0:
            monitor.log_status(f"Processed {video_idx + 1}/{len(video_paths)} videos")
    
    monitor.log_status("Worker finished")


def main():
    parser = argparse.ArgumentParser(description="Generate conversation summaries using multi-GPU inference")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files and transcripts")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="HuggingFace model name")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5", help="Comma-separated GPU IDs")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos that already have summaries")
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Find all video files with transcripts
    video_files = []
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        for video_path in video_dir.glob(f"*{ext}"):
            transcript_path = video_path.with_suffix(".json")
            if transcript_path.exists():
                video_files.append(video_path)
    
    if not video_files:
        print(f"No video files with transcripts found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} videos with transcripts")
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    
    # Distribute videos across GPUs (round-robin)
    gpu_video_assignments = [[] for _ in range(num_gpus)]
    for idx, video_path in enumerate(video_files):
        gpu_idx = idx % num_gpus
        gpu_video_assignments[gpu_idx].append(video_path)
    
    # Print distribution
    for gpu_idx, videos in enumerate(gpu_video_assignments):
        print(f"GPU {gpu_ids[gpu_idx]}: {len(videos)} videos")
    
    # Launch workers
    mp.set_start_method("spawn", force=True)
    processes = []
    
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        video_subset = gpu_video_assignments[gpu_idx]
        if not video_subset:
            continue
        
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, video_subset, args.model_name, args.skip_existing)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    print("All workers finished!")


if __name__ == "__main__":
    main()