"""
Loader for psychotherapy data from the openpsytslm pipeline.

This loader reads:
1. data_model.yaml - maps videos to transcripts and AU CSV files
2. Combined description JSON files - from combine_transcripts_with_time_series_descriptions.py
3. OpenFace AU CSV files - for extracting actual time series data

The loader creates training samples compatible with the opentslm framework.
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import pickle


def load_psychotherapy_cot_splits(
    data_model_path: str,
    combined_dir: str,
    train_videos: List[str] = None,
    val_videos: List[str] = None,
    test_videos: List[str] = None,
    max_samples: int = None,
    feature_columns: List[str] = None,
    max_seq_length: int = 4096
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load psychotherapy data and create train/val/test splits by video.
    
    Args:
        data_model_path: Path to data_model.yaml (from map_speaker_to_aus.py)
        combined_dir: Directory containing {video_id}_combined.json files
        train_videos: List of video IDs for training (if None, uses first 70%)
        val_videos: List of video IDs for validation (if None, uses next 15%)
        test_videos: List of video IDs for test (if None, uses last 15%)
        max_samples: Maximum samples per split (for debugging; None = no limit)
        feature_columns: List of AU column names to extract (default: all AU*_r columns)
                        Example: ['AU04_r'] for debug mode with single feature
        max_seq_length: Maximum sequence length after downsampling (default: 4096)
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples) as lists of dicts
    """
    # Disk caching
    cache_key = f"{data_model_path}_{combined_dir}_{max_samples}_{feature_columns}_{max_seq_length}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = Path(f"psychotherapy_cache_{cache_hash}.pkl")
    
    # Try to load from cache
    if cache_file.exists():
        print(f"[psychotherapy_loader] Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"[psychotherapy_loader] Loading data model from {data_model_path}")
    with open(data_model_path) as f:
        data_model = yaml.safe_load(f)
    
    # Get all video IDs
    all_video_ids = list(data_model.keys())
    
    # Create default splits if not provided (70/15/15)
    if train_videos is None and val_videos is None and test_videos is None:
        n_total = len(all_video_ids)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_videos = all_video_ids[:n_train]
        val_videos = all_video_ids[n_train:n_train + n_val]
        test_videos = all_video_ids[n_train + n_val:]
    
    train_videos_set = set(train_videos or [])
    val_videos_set = set(val_videos or [])
    test_videos_set = set(test_videos or [])
    
    # Load combined description files
    combined_dir_path = Path(combined_dir)
    combined_data = {}
    
    print(f"[psychotherapy_loader] Loading combined descriptions from {combined_dir}")
    for json_file in combined_dir_path.glob("*_combined.json"):
        video_id = json_file.stem.replace('_combined', '')
        with open(json_file, 'r', encoding='utf-8') as f:
            combined_data[video_id] = json.load(f)
    
    print(f"[psychotherapy_loader] Loaded {len(combined_data)} combined description files")
    
    # Process splits
    train_samples = _process_split(
        data_model, combined_data, train_videos_set,
        max_samples, feature_columns, max_seq_length
    )
    val_samples = _process_split(
        data_model, combined_data, val_videos_set,
        max_samples, feature_columns, max_seq_length
    )
    test_samples = _process_split(
        data_model, combined_data, test_videos_set,
        max_samples, feature_columns, max_seq_length
    )
    
    print(f"[psychotherapy_loader] Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    result = (train_samples, val_samples, test_samples)
    
    # Save to cache
    print(f"[psychotherapy_loader] Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def _process_split(
    data_model: Dict,
    combined_data: Dict[str, List[Dict]],
    video_ids: set,
    max_samples: int = None,
    feature_columns: List[str] = None,
    max_seq_length: int = 4096
) -> List[Dict]:
    """
    Process all turns for videos in the given split.
    
    Args:
        data_model: Data model dict from data_model.yaml
        combined_data: Dict mapping video_id -> list of turn dicts from combined JSON
        video_ids: Set of video IDs in this split
        max_samples: Optional limit for debugging
        feature_columns: Optional list of AU columns to extract
        max_seq_length: Maximum sequence length
    
    Returns:
        List of sample dicts ready for training
    """
    samples = []
    csv_cache = {}  # Cache loaded CSVs
    
    for video_id in video_ids:
        # Early exit if we have enough samples
        if max_samples is not None and len(samples) >= max_samples:
            print(f"[debug] Reached max_samples={max_samples}, stopping early")
            break
        
        # Check if we have combined data for this video
        if video_id not in combined_data:
            print(f"[warn] No combined data found for video: {video_id}")
            continue
        
        # Check if video exists in data model
        if video_id not in data_model:
            print(f"[warn] Video not found in data_model: {video_id}")
            continue
        
        video_info = data_model[video_id]
        turns = combined_data[video_id]
        
        # Get AU CSV paths
        speaker_1_csv = Path(video_info['AUs_speaker_1'])
        speaker_2_csv = Path(video_info['AUs_speaker_2'])
        
        if not speaker_1_csv.exists() or not speaker_2_csv.exists():
            print(f"[warn] AU CSVs not found for video: {video_id}")
            continue
        
        # Load or get from cache
        cache_key = (str(speaker_1_csv), str(speaker_2_csv))
        if cache_key in csv_cache:
            speaker_1_df, speaker_2_df = csv_cache[cache_key]
        else:
            try:
                print(f"[debug] Loading AU CSVs for {video_id}")
                speaker_1_df = pd.read_csv(speaker_1_csv, engine='python')
                speaker_2_df = pd.read_csv(speaker_2_csv, engine='python')
                csv_cache[cache_key] = (speaker_1_df, speaker_2_df)
            except Exception as e:
                print(f"[error] Loading CSVs for {video_id}: {e}")
                continue
        
        # Process each turn
        for turn in turns:
            if max_samples is not None and len(samples) >= max_samples:
                break
            
            sample = _extract_single_window(
                speaker_1_df, speaker_2_df,
                turn, video_id,
                video_info.get('speaker_mapping', {}),
                feature_columns,
                max_seq_length
            )
            
            if sample is not None:
                samples.append(sample)
    
    print(f"[info] Created {len(samples)} samples from {len(video_ids)} videos")
    return samples


def _extract_single_window(
    speaker_1_df: pd.DataFrame,
    speaker_2_df: pd.DataFrame,
    turn: Dict,
    video_id: str,
    speaker_mapping: Dict,
    feature_columns: List[str] = None,
    max_seq_length: int = 4096
) -> Dict:
    """
    Extract a single time window from speaker AU data.
    
    Args:
        speaker_1_df: Speaker 1 OpenFace CSV DataFrame
        speaker_2_df: Speaker 2 OpenFace CSV DataFrame
        turn: Turn dict from combined JSON with keys:
              video_id, turn_index, speaker_id, start_ms, end_ms, duration_ms,
              original_timeseries_description, original_summary, combined_description
        video_id: Video identifier
        speaker_mapping: Dict mapping speaker_1/speaker_2 to left/right
        feature_columns: Optional list of AU column names
        max_seq_length: Maximum sequence length
    
    Returns:
        Sample dict or None if extraction fails
    """
    # Normalize column names
    speaker_1_df.columns = speaker_1_df.columns.str.strip()
    speaker_2_df.columns = speaker_2_df.columns.str.strip()
    
    # Find timestamp column
    timestamp_col = None
    for col in speaker_1_df.columns:
        if col.lower() == 'timestamp':
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print(f"[error] No 'timestamp' column found for {video_id}")
        return None
    
    # Convert timestamps to numeric (in seconds)
    speaker_1_df[timestamp_col] = pd.to_numeric(speaker_1_df[timestamp_col], errors='coerce')
    speaker_2_df[timestamp_col] = pd.to_numeric(speaker_2_df[timestamp_col], errors='coerce')
    
    # Drop NaN timestamps
    try:
        speaker_1_df = speaker_1_df.dropna(subset=[timestamp_col])
        speaker_2_df = speaker_2_df.dropna(subset=[timestamp_col])
    except Exception as e:
        print(f"[error] Failed to process data for {video_id}: {e}")
        return None
    
    if speaker_1_df.empty or speaker_2_df.empty:
        return None
    
    # Convert milliseconds to seconds
    start_time = turn['start_ms'] / 1000.0
    end_time = turn['end_ms'] / 1000.0
    
    # Determine which AU columns to extract
    if feature_columns is not None:
        au_cols = [c for c in feature_columns if c in speaker_1_df.columns and c in speaker_2_df.columns]
        if len(au_cols) < len(feature_columns):
            missing_cols = set(feature_columns) - set(au_cols)
            print(f"[warn] Missing columns for {video_id}: {missing_cols}")
    else:
        # Default: all AU regression columns
        au_cols = [c for c in speaker_1_df.columns if 'AU' in c and '_r' in c]
    
    if not au_cols:
        print(f"[warn] No AU columns found for {video_id}")
        return None
    
    # Extract time windows
    try:
        speaker_1_window = speaker_1_df[
            (speaker_1_df[timestamp_col] >= start_time) &
            (speaker_1_df[timestamp_col] < end_time)
        ]
        speaker_2_window = speaker_2_df[
            (speaker_2_df[timestamp_col] >= start_time) &
            (speaker_2_df[timestamp_col] < end_time)
        ]
    except Exception as e:
        print(f"[error] Failed to extract window for {video_id}: {e}")
        return None
    
    # Validate sufficient data points
    if len(speaker_1_window) < 10 or len(speaker_2_window) < 10:
        print(f"[warn] Insufficient data for {video_id} turn {turn['turn_index']}: "
              f"speaker_1={len(speaker_1_window)}, speaker_2={len(speaker_2_window)}")
        return None
    
    # Extract AU vectors for both speakers
    speaker_1_au_vectors = {}
    speaker_1_au_stats = {}
    
    try:
        for au_col in au_cols:
            au_signal = speaker_1_window[au_col].to_numpy()
            au_mean = float(au_signal.mean())
            au_std = float(au_signal.std())
            
            speaker_1_au_vectors[au_col] = au_signal.tolist()
            speaker_1_au_stats[au_col] = {"mean": au_mean, "std": au_std}
    except Exception as e:
        print(f"[error] Failed to extract speaker 1 AUs for {video_id}: {e}")
        return None
    
    speaker_2_au_vectors = {}
    speaker_2_au_stats = {}
    
    try:
        for au_col in au_cols:
            au_signal = speaker_2_window[au_col].to_numpy()
            au_mean = float(au_signal.mean())
            au_std = float(au_signal.std())
            
            speaker_2_au_vectors[au_col] = au_signal.tolist()
            speaker_2_au_stats[au_col] = {"mean": au_mean, "std": au_std}
    except Exception as e:
        print(f"[error] Failed to extract speaker 2 AUs for {video_id}: {e}")
        return None
    
    # Downsample if necessary
    sample_au = list(speaker_1_au_vectors.values())[0] if speaker_1_au_vectors else []
    actual_length = len(sample_au)
    
    if actual_length > max_seq_length:
        print(f"⚠️  Downsampling {video_id} turn {turn['turn_index']} from {actual_length} to {max_seq_length}")
        
        indices = np.linspace(0, actual_length - 1, max_seq_length, dtype=int)
        
        # Downsample speaker 1
        for au_col in speaker_1_au_vectors:
            original_signal = np.array(speaker_1_au_vectors[au_col])
            downsampled_signal = original_signal[indices]
            speaker_1_au_vectors[au_col] = downsampled_signal.tolist()
            speaker_1_au_stats[au_col]["mean"] = float(downsampled_signal.mean())
            speaker_1_au_stats[au_col]["std"] = float(downsampled_signal.std())
        
        # Downsample speaker 2
        for au_col in speaker_2_au_vectors:
            original_signal = np.array(speaker_2_au_vectors[au_col])
            downsampled_signal = original_signal[indices]
            speaker_2_au_vectors[au_col] = downsampled_signal.tolist()
            speaker_2_au_stats[au_col]["mean"] = float(downsampled_signal.mean())
            speaker_2_au_stats[au_col]["std"] = float(downsampled_signal.std())
    
    # Create sample dict
    # Note: We use generic speaker_1/speaker_2 terminology here
    # The dataset class should map these to therapist/patient based on context
    sample = {
        "video_id": video_id,
        "turn_index": turn['turn_index'],
        "speaker_id": turn['speaker_id'],
        "window_start": start_time,
        "window_end": end_time,
        "start_ms": turn['start_ms'],
        "end_ms": turn['end_ms'],
        "speaker_1_au_vectors": speaker_1_au_vectors,
        "speaker_1_au_stats": speaker_1_au_stats,
        "speaker_2_au_vectors": speaker_2_au_vectors,
        "speaker_2_au_stats": speaker_2_au_stats,
        "au_columns": au_cols,
        "original_summary": turn['original_summary'],
        "original_timeseries_description": turn['original_timeseries_description'],
        "answer": turn['combined_description'],  # This is the target
        "labels": {},  # Placeholder for session-level metadata
        "baseline": {}  # Placeholder for baseline data
    }
    
    return sample


if __name__ == "__main__":
    # Test the loader
    train, val, test = load_psychotherapy_cot_splits(
        data_model_path="data_model.yaml",
        combined_dir="results/combined/",
        max_samples=5
    )
    
    print(f"\nTrain samples: {len(train)}")
    print(f"Val samples: {len(val)}")
    print(f"Test samples: {len(test)}")
    
    if train:
        print("\nSample from training set:")
        sample = train[0]
        print(f"Video ID: {sample['video_id']}")
        print(f"Turn index: {sample['turn_index']}")
        print(f"Speaker: {sample['speaker_id']}")
        print(f"Time window: {sample['start_ms']}ms - {sample['end_ms']}ms")
        print(f"AU columns: {sample['au_columns'][:5]}...")  # First 5
        print(f"Answer: {sample['answer'][:100]}...")  # First 100 chars
