"""
Parallel YouTube video downloader for zoomin dataset.

Usage:
    python download_youtube_videos.py --output-dir /path/to/videos --workers 8

This script:
  - Filters zoomin_info.csv for view=="on" and participants==2.0
  - Downloads videos in parallel using multiprocessing
  - Names files by YouTube video ID (e.g., Ea8dVOKS6aM.mp4)
  - Skips already downloaded videos
  - Reports progress and errors
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
import yt_dlp


def download_video(row, output_dir, skip_existing=True):
    """
    Download a single video from a DataFrame row.
    
    Args:
        row: pandas Series with 'vid_id' and 'vid_url'
        output_dir: Path object for output directory
        skip_existing: bool, skip if file exists
        
    Returns:
        dict with status info
    """
    vid_id = row['vid_id']
    vid_url = row['vid_url']
    output_path = output_dir / f"{vid_id}.mp4"
    
    # Skip if already exists
    if skip_existing and output_path.exists():
        return {
            'vid_id': vid_id,
            'status': 'skipped',
            'message': 'File already exists'
        }
    
    # yt-dlp options
    # Use pre-merged formats to avoid ffmpeg dependency:
    # - 'best[ext=mp4]' gets best quality single file (video+audio already combined)
    # - 'best' fallback for any available format
    # - '18' = 360p mp4, '22' = 720p mp4 (classic pre-merged YouTube formats)
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # No merging needed = no ffmpeg
        'outtmpl': str(output_path),
        'quiet': True,
        'no_warnings': True,
        'retries': 10,
        'fragment_retries': 10,
        # Headers to avoid 403 errors
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.youtube.com/'
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([vid_url])
        return {
            'vid_id': vid_id,
            'status': 'success',
            'message': 'Downloaded successfully'
        }
    except Exception as e:
        return {
            'vid_id': vid_id,
            'status': 'error',
            'message': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube videos from zoomin dataset in parallel'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='videos',
        help='Directory to save downloaded videos (default: videos/)'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='external/facet/dataset/zoomin_info.csv',
        help='Path to CSV file (default: external/facet/dataset/zoomin_info.csv)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip already downloaded videos (default: True)'
    )
    parser.add_argument(
        '--view',
        type=str,
        default='on',
        help='Filter by view type (default: "on")'
    )
    parser.add_argument(
        '--participants',
        type=float,
        default=2.0,
        help='Filter by number of participants (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Set number of workers - respect SLURM environment if available
    if args.workers:
        workers = args.workers
    elif os.environ.get('SLURM_CPUS_PER_TASK'):
        workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        print(f"Detected SLURM environment (SLURM_CPUS_PER_TASK={workers})")
    else:
        workers = cpu_count()
    print(f"Using {workers} parallel workers")
    
    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load and filter CSV
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Apply filters
    df_filtered = df[
        (df["view"] == args.view) & 
        (df["participants"] == args.participants)
    ]
    
    print(f"Total videos in dataset: {len(df)}")
    print(f"Filtered videos (view={args.view}, participants={args.participants}): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("No videos match the filter criteria. Exiting.")
        sys.exit(0)
    
    # Prepare download function with output_dir bound
    download_func = partial(
        download_video, 
        output_dir=output_dir, 
        skip_existing=args.skip_existing
    )
    
    # Download in parallel
    print(f"\nStarting parallel download with {workers} workers...")
    print("="*60)
    
    with Pool(processes=workers) as pool:
        results = []
        for result in pool.imap_unordered(download_func, [row for _, row in df_filtered.iterrows()]):
            results.append(result)
            status_symbol = {
                'success': '✓',
                'skipped': '○',
                'error': '✗'
            }.get(result['status'], '?')
            print(f"[{status_symbol}] {result['vid_id']:15s} - {result['status']:10s} - {result['message']}")
    
    # Summary
    print("="*60)
    print("\nDownload Summary:")
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"  Total:      {len(results)}")
    print(f"  Success:    {success_count}")
    print(f"  Skipped:    {skipped_count}")
    print(f"  Errors:     {error_count}")
    
    if error_count > 0:
        print("\nFailed videos:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['vid_id']}: {r['message']}")
    
    print(f"\nVideos saved to: {output_dir}")


if __name__ == '__main__':
    main()