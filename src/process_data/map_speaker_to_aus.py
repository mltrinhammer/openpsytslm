#!/usr/bin/env python3
"""
Map speakers from transcription to AU CSV files (left/right).

This script determines which AU CSV file (left or right) corresponds to which speaker
(speaker_1 or speaker_2) from the transcription JSON, then outputs a data model YAML
that links videos, transcriptions, and AU files.

Strategy:
- For each video, we have 2 AU CSV files (left and right)
- We have 1 transcription JSON with speaker_1 and speaker_2
- We need to determine which speaker maps to which side
- This is done by interpreting mouth activitation during speaking intervals
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd
import numpy as np


def load_transcription(json_path: Path) -> List[Dict]:
    """Load transcription JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_au_csv(csv_path: Path) -> pd.DataFrame:
    """Load AU CSV file."""
    return pd.read_csv(csv_path)


def get_speaking_intervals(transcription: List[Dict], speaker: str) -> List[Tuple[float, float]]:
    """
    Extract time intervals when a specific speaker is speaking.
    
    Args:
        transcription: List of transcription segments
        speaker: Speaker label (e.g., 'speaker_1')
    
    Returns:
        List of (start, end) time tuples in seconds
    """
    intervals = []
    for segment in transcription:
        if segment.get("speaker") == speaker:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            intervals.append((start, end))
    return intervals


def calculate_au_activity(au_df: pd.DataFrame, intervals: List[Tuple[float, float]], 
                          fps: float = 30.0) -> float:
    """
    Calculate AU activity score during speaking intervals.
    
    The speaker's face should show more AU activity (especially mouth AUs) when speaking.
    
    Args:
        au_df: DataFrame with AU values
        intervals: List of (start, end) time intervals in seconds
        fps: Frames per second (default 30)
    
    Returns:
        Mean AU activity score during intervals
    """
    if au_df.empty or not intervals:
        return 0.0
    
    # Focus on mouth-related AUs (typically AU10, AU12, AU20, AU25, AU26, AU27)
    # These are more active during speech
    mouth_aus = []
    for col in au_df.columns:
        # Look for AU columns (typically named like AU1, AU2, etc.)
        if col.startswith('AU') or col.startswith('au'):
            # Extract AU number
            try:
                au_num = int(''.join(filter(str.isdigit, col)))
                # Mouth-related AUs: 10, 12, 14, 15, 17, 20, 23, 25, 26, 27
                if au_num in [10, 12, 14, 15, 17, 20, 23, 25, 26, 27]:
                    mouth_aus.append(col)
            except ValueError:
                continue
    
    if not mouth_aus:
        # Fallback: use all AU columns
        mouth_aus = [col for col in au_df.columns if col.startswith(('AU', 'au'))]
    
    if not mouth_aus:
        return 0.0
    
    # Calculate activity during speaking intervals
    activity_scores = []
    
    for start_sec, end_sec in intervals:
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Ensure frames are within bounds
        start_frame = max(0, min(start_frame, len(au_df) - 1))
        end_frame = max(0, min(end_frame, len(au_df)))
        
        if start_frame >= end_frame:
            continue
        
        # Extract AU values during this interval
        interval_aus = au_df.iloc[start_frame:end_frame][mouth_aus]
        
        # Calculate mean absolute AU values (activity)
        mean_activity = interval_aus.abs().mean().mean()
        activity_scores.append(mean_activity)
    
    return np.mean(activity_scores) if activity_scores else 0.0


def match_speaker_to_side(transcription: List[Dict], 
                          left_au_path: Path, 
                          right_au_path: Path,
                          fps: float = 30.0) -> Dict[str, str]:
    """
    Determine which speaker corresponds to which side (left/right).
    
    Strategy: The speaker's AU activity should be higher on their side during speaking.
    
    Args:
        transcription: Transcription data
        left_au_path: Path to left AU CSV
        right_au_path: Path to right AU CSV
        fps: Video frames per second
    
    Returns:
        Dict mapping speaker labels to side labels, e.g.:
        {'speaker_1': 'left', 'speaker_2': 'right'}
    """
    # Load AU files
    try:
        left_au = load_au_csv(left_au_path)
        right_au = load_au_csv(right_au_path)
    except Exception as e:
        print(f"Warning: Could not load AU files: {e}")
        # Default mapping
        return {'speaker_1': 'left', 'speaker_2': 'right'}
    
    # Get speaking intervals for each speaker
    speaker1_intervals = get_speaking_intervals(transcription, 'speaker_1')
    speaker2_intervals = get_speaking_intervals(transcription, 'speaker_2')
    
    # Calculate AU activity for each combination
    speaker1_left_activity = calculate_au_activity(left_au, speaker1_intervals, fps)
    speaker1_right_activity = calculate_au_activity(right_au, speaker1_intervals, fps)
    
    speaker2_left_activity = calculate_au_activity(left_au, speaker2_intervals, fps)
    speaker2_right_activity = calculate_au_activity(right_au, speaker2_intervals, fps)
    
    # Decision logic:
    # - If speaker_1 has higher activity on left, then speaker_1 = left
    # - If speaker_1 has higher activity on right, then speaker_1 = right
    
    speaker1_prefers_left = speaker1_left_activity > speaker1_right_activity
    speaker2_prefers_left = speaker2_left_activity > speaker2_right_activity
    
    # Use the stronger signal
    if speaker1_prefers_left and not speaker2_prefers_left:
        return {'speaker_1': 'left', 'speaker_2': 'right'}
    elif not speaker1_prefers_left and speaker2_prefers_left:
        return {'speaker_1': 'right', 'speaker_2': 'left'}
    elif speaker1_prefers_left:
        # Both prefer left, use the one with stronger preference
        s1_diff = speaker1_left_activity - speaker1_right_activity
        s2_diff = speaker2_left_activity - speaker2_right_activity
        if s1_diff > s2_diff:
            return {'speaker_1': 'left', 'speaker_2': 'right'}
        else:
            return {'speaker_1': 'right', 'speaker_2': 'left'}
    else:
        # Both prefer right, use the one with stronger preference
        s1_diff = speaker1_right_activity - speaker1_left_activity
        s2_diff = speaker2_right_activity - speaker2_left_activity
        if s1_diff > s2_diff:
            return {'speaker_1': 'right', 'speaker_2': 'left'}
        else:
            return {'speaker_1': 'left', 'speaker_2': 'right'}


def process_video(video_path: Path, 
                  transcript_dir: Path, 
                  au_dir: Path,
                  fps: float = 30.0) -> Optional[Dict]:
    """
    Process a single video and create data model entry.
    
    Args:
        video_path: Path to source video
        transcript_dir: Directory containing transcription JSONs (flat or nested)
        au_dir: Directory containing AU CSV files
        fps: Video frame rate
    
    Returns:
        Data model dict for this video, or None if files missing
    """
    video_stem = video_path.stem
    
    # Find transcription file - try both flat and nested structure
    # First try nested: transcript_dir/video_stem/results_{video_stem}.json
    transcript_path = transcript_dir / video_stem / f"results_{video_stem}.json"
    
    # If not found, try flat structure: transcript_dir/results_{video_stem}.json
    if not transcript_path.exists():
        transcript_path = transcript_dir / f"results_{video_stem}.json"
    
    # Try without "results_" prefix
    if not transcript_path.exists():
        transcript_path = transcript_dir / f"{video_stem}.json"
    
    if not transcript_path.exists():
        print(f"Warning: Transcription not found for {video_stem}")
        return None
    
    # Find AU files
    left_au_path = au_dir / f"{video_stem}_left.csv"
    right_au_path = au_dir / f"{video_stem}_right.csv"
    
    if not left_au_path.exists() or not right_au_path.exists():
        print(f"Warning: AU files not found for {video_stem}")
        return None
    
    # Load transcription
    try:
        transcription = load_transcription(transcript_path)
    except Exception as e:
        print(f"Warning: Could not load transcription for {video_stem}: {e}")
        return None
    
    # Match speakers to sides
    speaker_mapping = match_speaker_to_side(transcription, left_au_path, right_au_path, fps)
    
    # Build data model entry
    entry = {
        'id': video_stem,
        'video_path': str(video_path.resolve()),
        'transcription_path': str(transcript_path.resolve()),
        'AUs_speaker_1': str((au_dir / f"{video_stem}_{speaker_mapping['speaker_1']}.csv").resolve()),
        'AUs_speaker_2': str((au_dir / f"{video_stem}_{speaker_mapping['speaker_2']}.csv").resolve()),
        'speaker_mapping': speaker_mapping
    }
    
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Map speakers from transcription to AU CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos
  python map_speaker_to_aus.py \\
    --video_dir results/source_recordings \\
    --transcript_dir results/transcripts \\
    --au_dir results/au_outputs \\
    --output data_model.yaml
  
  # Specify FPS if different from default (30)
  python map_speaker_to_aus.py \\
    --video_dir results/source_recordings \\
    --transcript_dir results/transcripts \\
    --au_dir results/au_outputs \\
    --output data_model.yaml \\
    --fps 25
        """
    )
    
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing source video files (*.mp4)")
    parser.add_argument("--transcript_dir", type=str, required=True,
                        help="Directory containing transcription JSON files (flat or nested structure)")
    parser.add_argument("--au_dir", type=str, required=True,
                        help="Directory containing AU CSV files")
    parser.add_argument("--output", type=str, default="data_model.yaml",
                        help="Output YAML file (default: data_model.yaml)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate (default: 30.0)")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    transcript_dir = Path(args.transcript_dir)
    au_dir = Path(args.au_dir)
    output_path = Path(args.output)
    
    # Validate directories
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)
    
    if not transcript_dir.exists():
        print(f"Error: Transcript directory not found: {transcript_dir}")
        sys.exit(1)
    
    if not au_dir.exists():
        print(f"Error: AU directory not found: {au_dir}")
        sys.exit(1)
    
    # Find all videos
    videos = list(video_dir.glob("*.mp4"))
    
    if not videos:
        print(f"No MP4 files found in {video_dir}")
        sys.exit(1)
    
    print(f"Found {len(videos)} video files")
    print(f"Processing with FPS: {args.fps}")
    print("-" * 70)
    
    # Process each video
    data_model = {}
    success_count = 0
    
    for video_path in sorted(videos):
        print(f"Processing: {video_path.name}")
        
        entry = process_video(video_path, transcript_dir, au_dir, args.fps)
        
        if entry:
            data_model[entry['id']] = {
                'video_path': entry['video_path'],
                'transcription_path': entry['transcription_path'],
                'AUs_speaker_1': entry['AUs_speaker_1'],
                'AUs_speaker_2': entry['AUs_speaker_2'],
                'speaker_mapping': entry['speaker_mapping']
            }
            success_count += 1
            print(f"  ✓ Mapped: speaker_1 → {entry['speaker_mapping']['speaker_1']}, "
                  f"speaker_2 → {entry['speaker_mapping']['speaker_2']}")
        else:
            print(f"  ✗ Skipped (missing files)")
    
    # Write output
    print("\n" + "=" * 70)
    print(f"Successfully processed: {success_count}/{len(videos)} videos")
    print(f"Writing data model to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_model, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ Data model saved: {output_path}")
    print("=" * 70)
    
    if success_count < len(videos):
        print(f"\nWarning: {len(videos) - success_count} video(s) skipped due to missing files")
        sys.exit(1)
    else:
        print("\n✓ All videos processed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
