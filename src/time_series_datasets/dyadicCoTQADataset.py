from typing import List, Tuple, Literal
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "external" / "opentslm" / "src"))

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset

# Import from local dyadloader
sys.path.append(str(Path(__file__).parent))
from dyadloader import load_dyadic_cot_splits


class DyadicCoTQADataset(QADataset):
    """
    Dyadic CoT QA Dataset for analyzing speaker facial AU time-series.
    
    This dataset processes synchronized facial action units (AUs) from two speakers
    during dyadic interactions. Each sample represents one speech turn with:
    - Original transcript summary as context
    - Synchronized AU time series for both speakers
    - Combined facial description as the target answer
    
    The dataset handles variable-length sequences within each turn by applying forward-fill
    padding to ensure all AU channels have identical length for tensor operations.
    
    Args:
        split: One of "train", "test", or "validation"
        EOS_TOKEN: End-of-sequence token for the language model
        data_model_path: Path to data_model.yaml (from map_speaker_to_aus.py)
        combined_dir: Directory containing {video_id}_combined.json files
        train_videos: List of video IDs for training split
        val_videos: List of video IDs for validation split
        test_videos: List of video IDs for test split
        format_sample_str: Whether to format samples as strings
        time_series_format_function: Optional function to format time series
        max_samples: Maximum number of samples to load (for debugging)
        feature_columns: List of AU column names to use (e.g., ['AU04_r'] for debugging)
        max_seq_length: Maximum sequence length; longer sequences are downsampled (default: 4096)
    """

    def __init__(self, 
                 split: Literal["train", "test", "validation"],
                 EOS_TOKEN: str,
                 data_model_path: str = None,
                 combined_dir: str = None,
                 train_videos: List[str] = None,
                 val_videos: List[str] = None,
                 test_videos: List[str] = None,
                 format_sample_str: bool = False, 
                 time_series_format_function=None,
                 max_samples: int = None,
                 feature_columns: List[str] = None,
                 max_seq_length: int = 4096):
        self.data_model_path = data_model_path or "data_model.yaml"
        self.combined_dir = combined_dir or "results/combined/"
        self.train_videos = train_videos
        self.val_videos = val_videos
        self.test_videos = test_videos
        self.max_samples = max_samples
        self.feature_columns = feature_columns
        self.max_seq_length = max_seq_length
        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN, 
            format_sample_str=format_sample_str, 
            time_series_format_function=time_series_format_function
        )

    def _load_splits(self) -> Tuple[List, List, List]:
        """Load train/val/test splits as plain Python lists."""
        train_list, val_list, test_list = load_dyadic_cot_splits(
            data_model_path=self.data_model_path,
            combined_dir=self.combined_dir,
            train_videos=self.train_videos,
            val_videos=self.val_videos,
            test_videos=self.test_videos,
            max_samples=self.max_samples,
            feature_columns=self.feature_columns,
            max_seq_length=self.max_seq_length
        )
        
        return train_list, val_list, test_list

    def _get_answer(self, row) -> str:
        """Get the answer from the combined description.
           """
        return row.get("answer", "")

    def _get_pre_prompt(self, row) -> str:
        """Generate the pre-prompt instruction with transcript summary."""
        original_summary = row.get("original_summary", "")
        speaker_id = row.get("speaker_id", "unknown")
        
        # Map speaker_1/speaker_2 to "Speaker 1"/"Speaker 2" for consistency with gold answers
        if speaker_id == "speaker_1":
            speaker_label = "Speaker 1"
        elif speaker_id == "speaker_2":
            speaker_label = "Speaker 2"
        else:
            speaker_label = speaker_id

        prompt = f"""You are describing a speech turn from a dyadic interaction between exactly two speakers: Speaker 1 and Speaker 2.

Speech content: {original_summary} (spoken by {speaker_label})

Instructions:
- Write ONE short paragraph combining the speech content with the facial expressions.
- ONLY refer to Speaker 1 and Speaker 2 - no other speakers exist.
- Briefly mention the most salient facial Action Units (AUs) - do not over-analyze.
- Do not speculate or add information not present in the data.

"""
        return prompt

    def _get_post_prompt(self, row) -> str:
        """Generate the post-prompt with answer formatting instruction."""
        return """Description: """

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into a list of TextTimeSeriesPrompt objects.
        Creates one prompt per AU sequence for both speakers, following HAR pattern.
        
        Note: The loader provides speaker_1 and speaker_2 data. We treat them generically
        here - the prompt text identifies them as speaker 1 and speaker 2.
        """
        # Get AU column names
        au_cols = row.get("au_columns", [])
        
        # Unpack data from dictionaries
        speaker_1_au_vectors = row["speaker_1_au_vectors"]
        speaker_1_au_stats = row["speaker_1_au_stats"]
        speaker_2_au_vectors = row["speaker_2_au_vectors"]
        speaker_2_au_stats = row["speaker_2_au_stats"]
        
        
        all_signals = []
        all_means = []
        all_stds = []
        all_labels = []
        
        # Add speaker 1 AUs
        for au_name in au_cols:
            signal = speaker_1_au_vectors[au_name]
            stats = speaker_1_au_stats[au_name]
            
            all_signals.append(signal)
            all_means.append(stats["mean"])
            all_stds.append(stats["std"])
            all_labels.append(f"speaker_1 for {au_name}")
        
        # Add speaker 2 AUs
        for au_name in au_cols:
            signal = speaker_2_au_vectors[au_name]
            stats = speaker_2_au_stats[au_name]
            
            all_signals.append(signal)
            all_means.append(stats["mean"])
            all_stds.append(stats["std"])
            all_labels.append(f"speaker_2 for {au_name}")

        max_length = max(len(sig) for sig in all_signals)
        
        padded_signals = []
        for signal in all_signals:
            if len(signal) < max_length:
                # Pad with the last value (forward fill)
                padded = signal + [signal[-1]] * (max_length - len(signal))
            else:
                padded = signal
            padded_signals.append(padded)
        
        # Create tensor from padded signals
        series = torch.tensor(padded_signals, dtype=torch.float32)
        
        # Handle NaN values (from failed face detection frames) by filling with 0
        if torch.isnan(series).any():
            nan_count = torch.isnan(series).sum().item()
            print(f"⚠️  Filling {nan_count} NaN values with 0 in sample from {row.get('video_id', 'unknown')}")
            series = torch.nan_to_num(series, nan=0.0)
        
        # Check for infinity values (still invalid)
        if torch.isinf(series).any():
            print(f"❌ Invalid data detected in Psychotherapy CoT sample")
            print(f"Row keys: {row.keys()}")
            print(f"Series shape: {series.shape}")
            print(f"Inf positions: {torch.isinf(series).nonzero()}")
            raise ValueError("Invalid data detected: contains Inf values")
        
        # Recalculate mean and std from the cleaned signal (matching ECG-QA approach)
        # This ensures we use clean statistics after NaN handling
        means_tensor = series.mean(dim=1, keepdim=True)
        stds_tensor = series.std(dim=1, keepdim=True)
        
        # Clamp stds to avoid division by zero (matching ECG-QA and HAR CoT)
        min_std = 1e-6
        stds_tensor = torch.clamp(stds_tensor, min=min_std)
        
        # Normalize: (x - mean) / std
        series_norm = (series - means_tensor) / stds_tensor
        
        # Convert means/stds back to lists for prompt text
        all_means = means_tensor.squeeze(1).tolist()
        all_stds = stds_tensor.squeeze(1).tolist()
        
        # Check for invalid data after normalization
        if torch.isnan(series_norm).any() or torch.isinf(series_norm).any():
            print(f"❌ NaN/Inf detected after normalization")
            print(f"Original series shape: {series.shape}")
            print(f"Means: {means_tensor.squeeze()}")
            print(f"Stds: {stds_tensor.squeeze()}")
            print(f"NaN positions: {torch.isnan(series_norm).nonzero()}")
            print(f"Inf positions: {torch.isinf(series_norm).nonzero()}")
            raise ValueError("NaN/Inf detected after normalization")
        
        # Create prompts (one per AU, using normalized tensor data)
        prompts = []
        au_idx = 0
        
        # Speaker 1 AUs
        for au_name in au_cols:
            mean = all_means[au_idx]
            std = all_stds[au_idx]
            normalized_series = series_norm[au_idx].tolist()
            
            text_prompt = f"[Speaker 1 {au_name}] mean={mean:.4f}, std={std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series))
            au_idx += 1
        
        # Speaker 2 AUs
        for au_name in au_cols:
            mean = all_means[au_idx]
            std = all_stds[au_idx]
            normalized_series = series_norm[au_idx].tolist()
            
            text_prompt = f"[Speaker 2 {au_name}] mean={mean:.4f}, std={std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, normalized_series))
            au_idx += 1
        
        return prompts

    def _format_sample(self, row):
        """Format the sample with additional metadata."""
        sample = super()._format_sample(row)
        sample["video_id"] = row["video_id"]
        sample["turn_index"] = row.get("turn_index")
        sample["speaker_id"] = row.get("speaker_id")
        sample["window_start"] = row["window_start"]
        sample["window_end"] = row["window_end"]
        sample["labels"] = row.get("labels", {})
        sample["baseline"] = row.get("baseline", {})
        sample["answer"] = row.get("answer", {})
        return sample


# Test the dataset
if __name__ == "__main__":
    dataset = DyadicCoTQADataset(
        split="train", 
        EOS_TOKEN=";", 
        data_model_path="data_model.yaml",
        combined_dir="results/combined/",
        max_samples=5
    )    
    print(len(dataset))
    # Show sample data
    if len(dataset) > 0:
        print("\n" + "="*50 + "\n")
        print("Sample data from training set:")
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Video ID:", sample.get("video_id"))
        print("Speaker ID:", sample.get("speaker_id"))
        print("Turn index:", sample.get("turn_index"))
        print("Window:", f"{sample.get('window_start'):.2f}s - {sample.get('window_end'):.2f}s")
        print("Answer:", sample.get("answer")[:100] + "..." if len(sample.get("answer", "")) > 100 else sample.get("answer"))
        print("\nFirst time series prompt:")
        print(sample["time_series_text"][0])
        print("First few values:", sample["time_series"][0][:10])
        print("\nSecond time series prompt:")
        print(sample["time_series_text"][1])
        print("First few values:", sample["time_series"][1][:10])
