"""
Validate AU CSV files against source videos.

Checks that the number of rows in AU CSV files matches the expected frame count
from the source video (fps * duration).
"""

import argparse
import csv
import sys
from pathlib import Path
import cv2


def get_video_info(video_path: Path) -> dict:
    """Get video metadata (fps, frame count, duration)."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "expected_rows": frame_count
    }


def count_csv_rows(csv_path: Path) -> int:
    """Count rows in CSV file (excluding header)."""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        # Skip header if present
        next(reader, None)
        return sum(1 for _ in reader)


def check_au_file(video_path: Path, csv_path: Path, tolerance: int = 0) -> dict:
    """
    Check if AU CSV file has expected number of rows.
    
    Args:
        video_path: Path to source video
        csv_path: Path to AU CSV file
        tolerance: Acceptable difference in row count (default: 0)
    
    Returns:
        dict with validation results
    """
    result = {
        "video": video_path.name,
        "csv": csv_path.name,
        "status": "unknown",
        "message": ""
    }
    
    try:
        # Get video info
        video_info = get_video_info(video_path)
        expected = video_info["expected_rows"]
        
        # Count CSV rows
        actual = count_csv_rows(csv_path)
        
        # Compare
        diff = abs(actual - expected)
        
        result["expected_rows"] = expected
        result["actual_rows"] = actual
        result["difference"] = actual - expected
        result["fps"] = video_info["fps"]
        result["duration"] = video_info["duration"]
        
        if diff <= tolerance:
            result["status"] = "OK"
            result["message"] = f"Row count matches (expected={expected}, actual={actual})"
        else:
            result["status"] = "MISMATCH"
            result["message"] = f"Row count mismatch: expected {expected}, got {actual} (diff={actual-expected})"
        
    except Exception as e:
        result["status"] = "ERROR"
        result["message"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate AU CSV files against source videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all AU files
  python check_aus.py --video_dir ./videos --output_dir ./au_outputs
  
  # Allow 1 frame difference
  python check_aus.py --video_dir ./videos --output_dir ./au_outputs --tolerance 1
  
  # Verbose output
  python check_aus.py --video_dir ./videos --output_dir ./au_outputs --verbose
        """
    )
    
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing source video files (*.mp4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing AU CSV files (*_left.csv, *_right.csv)")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Acceptable difference in row count (default: 0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information for all files")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Find all videos
    videos = list(video_dir.glob("*.mp4"))
    
    if not videos:
        print(f"No MP4 files found in {video_dir}")
        sys.exit(1)
    
    print(f"Found {len(videos)} video files")
    print(f"Checking AU CSV files in {output_dir}")
    print(f"Tolerance: {args.tolerance} frame(s)")
    print("-" * 70)
    
    # Check each video's AU files
    results = []
    total_ok = 0
    total_mismatch = 0
    total_error = 0
    total_missing = 0
    
    for video_path in sorted(videos):
        stem = video_path.stem
        
        # Check left and right CSV files
        for side in ["left", "right"]:
            csv_path = output_dir / f"{stem}_{side}.csv"
            
            if not csv_path.exists():
                results.append({
                    "video": video_path.name,
                    "csv": f"{stem}_{side}.csv",
                    "status": "MISSING",
                    "message": "CSV file not found"
                })
                total_missing += 1
                continue
            
            result = check_au_file(video_path, csv_path, args.tolerance)
            results.append(result)
            
            if result["status"] == "OK":
                total_ok += 1
            elif result["status"] == "MISMATCH":
                total_mismatch += 1
            else:
                total_error += 1
    
    # Print results
    if args.verbose:
        print("\nDetailed Results:")
        print("=" * 70)
        for r in results:
            status_symbol = {
                "OK": "✓",
                "MISMATCH": "✗",
                "MISSING": "⚠",
                "ERROR": "⚠"
            }.get(r["status"], "?")
            
            print(f"\n{status_symbol} {r['csv']}")
            print(f"  Video: {r['video']}")
            print(f"  Status: {r['status']}")
            print(f"  {r['message']}")
            
            if "expected_rows" in r:
                print(f"  Expected rows: {r['expected_rows']}")
                print(f"  Actual rows: {r['actual_rows']}")
                print(f"  FPS: {r['fps']:.2f}")
                print(f"  Duration: {r['duration']:.2f}s")
    else:
        # Print only problems
        problems = [r for r in results if r["status"] != "OK"]
        if problems:
            print("\nProblems Found:")
            print("=" * 70)
            for r in problems:
                status_symbol = {
                    "MISMATCH": "✗",
                    "MISSING": "⚠",
                    "ERROR": "⚠"
                }.get(r["status"], "?")
                print(f"{status_symbol} {r['csv']}: {r['message']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files checked: {len(results)}")
    print(f"  ✓ OK:          {total_ok}")
    print(f"  ✗ Mismatch:    {total_mismatch}")
    print(f"  ⚠ Missing:     {total_missing}")
    print(f"  ⚠ Error:       {total_error}")
    print("=" * 70)
    
    # Exit code
    if total_mismatch > 0 or total_error > 0 or total_missing > 0:
        print("\n⚠ Validation FAILED")
        sys.exit(1)
    else:
        print("\n✓ All files validated successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
