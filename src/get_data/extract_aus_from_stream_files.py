"""
Test script to load and extract Action Units (AUs) from OpenFace2 .stream files
Processes multiple session directories
"""
import sys
import os
import numpy as np
import argparse

# Add the documentation/read_files directory to the path to import the utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'documentation', 'read_files'))

from ssi_stream_utils import Stream


def load_column_names(openface_txt_path):
    """Load column names from the video.openface2.txt file"""
    with open(openface_txt_path, 'r') as f:
        columns = [line.strip() for line in f.readlines()]
    return columns


def get_au_indices(columns):
    """Find the indices of frame, timestamp, and AU columns ending with _r (regression values)"""
    indices = []
    names = []
    
    # Add frame and timestamp first
    for idx, col in enumerate(columns):
        if col == 'frame' or col == 'timestamp':
            indices.append(idx)
            names.append(col)
    
    # Then add AU columns
    for idx, col in enumerate(columns):
        if col.startswith('AU') and col.endswith('_r'):
            indices.append(idx)
            names.append(col)
    
    return indices, names


def extract_aus_from_stream(stream_path, au_indices, name):
    """Load a .stream file and extract only the AU columns"""
    
    print(f"  Extracting AUs from {name}...", end=' ')
    
    try:
        # Load the stream
        stream = Stream().load(stream_path)
        
        # Extract only AU columns
        if stream.data is not None:
            au_data = stream.data[:, au_indices]
            print(f"✓ ({au_data.shape[0]} samples)")
            return au_data
        else:
            print("✗ No data found")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def process_session(session_dir, au_indices, au_names):
    """Process a single session directory"""
    session_name = os.path.basename(session_dir)
    print(f"\nProcessing session {session_name}:")
    
    expert_path = os.path.join(session_dir, "expert.video.openface2.stream")
    novice_path = os.path.join(session_dir, "novice.video.openface2.stream")
    
    # Check if files exist
    if not os.path.exists(expert_path):
        print(f"  ✗ Expert file not found, skipping")
        return False
    
    if not os.path.exists(novice_path):
        print(f"  ✗ Novice file not found, skipping")
        return False
    
    # Extract AUs from both streams
    expert_aus = extract_aus_from_stream(expert_path, au_indices, "expert")
    novice_aus = extract_aus_from_stream(novice_path, au_indices, "novice")
    
    if expert_aus is None or novice_aus is None:
        print(f"  ✗ Failed to extract AUs")
        return False
    
    # Save the AU data to CSV files
    expert_save_path = os.path.join(session_dir, "expert_aus.csv")
    novice_save_path = os.path.join(session_dir, "novice_aus.csv")
    
    # Save with column headers
    header = ','.join(au_names)
    np.savetxt(expert_save_path, expert_aus, delimiter=',', header=header, comments='', fmt='%.6f')
    np.savetxt(novice_save_path, novice_aus, delimiter=',', header=header, comments='', fmt='%.6f')
    
    print(f"  ✓ Saved: expert_aus.csv ({expert_aus.shape[0]} x {expert_aus.shape[1]})")
    print(f"  ✓ Saved: novice_aus.csv ({novice_aus.shape[0]} x {novice_aus.shape[1]})")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Action Units from OpenFace2 stream files')
    parser.add_argument('--dir', type=str, default=None, 
                        help='Directory containing session subdirectories (e.g., 026, 027, etc.)')
    
    args = parser.parse_args()
    
    # Determine the base directory
    results_dir = os.path.dirname(__file__)
    
    if args.dir:
        base_dir = os.path.abspath(args.dir)
    else:
        base_dir = results_dir
    
    doc_dir = os.path.join(results_dir, "documentation")
    columns_path = os.path.join(doc_dir, "video.openface2.txt")
    
    # Check if column names file exists
    if not os.path.exists(columns_path):
        print(f"Error: Column names file not found at {columns_path}")
        sys.exit(1)
    
    # Load column names and find AU indices
    print("Loading column names from video.openface2.txt...")
    columns = load_column_names(columns_path)
    au_indices, au_names = get_au_indices(columns)
    
    num_aus = len([n for n in au_names if n.startswith('AU')])
    print(f"Found {num_aus} AU columns (regression values ending with '_r')")
    print(f"Extracting columns: frame, timestamp, and {num_aus} AUs")
    print(f"Column names: {au_names}\n")
    
    # Find all session directories (numeric names)
    session_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            session_dirs.append(item_path)
    
    session_dirs.sort()
    
    if not session_dirs:
        print(f"No session directories found in {base_dir}")
        print("Looking for directories with numeric names (e.g., 026, 027, etc.)")
        sys.exit(1)
    
    print(f"Found {len(session_dirs)} session directories")
    print("="*60)
    
    # Process each session
    success_count = 0
    failed_sessions = []
    
    for session_dir in session_dirs:
        if process_session(session_dir, au_indices, au_names):
            success_count += 1
        else:
            failed_sessions.append(os.path.basename(session_dir))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total sessions: {len(session_dirs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_sessions)}")
    
    if failed_sessions:
        print(f"Failed sessions: {', '.join(failed_sessions)}")
    
    print("\nDone! Action Units have been successfully extracted and saved.")
