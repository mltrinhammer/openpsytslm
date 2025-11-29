import argparse
import os
import sys
import csv
import shutil
import tempfile
import torch
import cv2
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
import traceback

# OpenFace Imports
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

# -----------------------------------------------------------------------------
# HELPER: TEMP FILE HANDLER
# -----------------------------------------------------------------------------
class TempImageHandler:
    """
    Manages a temporary directory and file for passing images to get_face().
    """
    def __init__(self):
        # We create a unique temp dir per process/thread to avoid collisions
        self.temp_dir = tempfile.mkdtemp(prefix=f"openface_proc_{os.getpid()}_")
        self.temp_img_path = os.path.join(self.temp_dir, "current_frame.jpg")

    def save_frame(self, frame):
        """Saves numpy frame to disk so OpenFace can read the path."""
        cv2.imwrite(self.temp_img_path, frame)
        return self.temp_img_path

    def cleanup(self):
        """Removes the temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_models(weights_root, device):
    """
    Loads models on the specific GPU assigned to this worker.
    """
    weights_dir = Path(weights_root) / 'weights'
    
    # Updated filenames based on your previous 'ls' output
    face_model_path = weights_dir / 'mobilenet0.25_Final.pth'
    mtl_model_path = weights_dir / 'stage2_epoch_7_loss_1.1606_acc_0.5589.pth'

    print(f"[{device}] Loading models...")
    
    # Initialize models on the specific device (e.g., 'cuda:0', 'cuda:1')
    face_detector = FaceDetector(model_path=str(face_model_path), device=device)
    multitask_model = MultitaskPredictor(model_path=str(mtl_model_path), device=device)

    return face_detector, multitask_model

# -----------------------------------------------------------------------------
# PROCESSING LOGIC
# -----------------------------------------------------------------------------
def process_frame(frame, face_detector, multitask_model, temp_handler):
    # 1. Save frame to temp file (Required by library)
    img_path = temp_handler.save_frame(frame)

    # 2. Detect Face
    try:
        cropped_face, dets = face_detector.get_face(img_path)
    except Exception:
        return False, None

    # 3. Validate the detection
    # FIX: We strictly check for None AND for empty arrays (size == 0)
    if cropped_face is None or dets is None or cropped_face.size == 0:
        return False, None

    # 4. Predict AUs
    try:
        _, _, au_output = multitask_model.predict(cropped_face)
        if isinstance(au_output, torch.Tensor):
            au_output = au_output.cpu().detach().numpy().flatten().tolist()
        return True, au_output
    except Exception as e:
        # Catch internal OpenFace errors to prevent worker death
        # print(f"Predict error: {e}") 
        return False, None
    
    return False, None

def process_video(video_path, output_dir, face_detector, multitask_model, temp_handler, worker_id):
    filename_stem = Path(video_path).stem
    
    # Define output paths
    left_csv_path = os.path.join(output_dir, f"{filename_stem}_left.csv")
    right_csv_path = os.path.join(output_dir, f"{filename_stem}_right.csv")

    # Skip if output already exists
    if os.path.exists(left_csv_path) and os.path.exists(right_csv_path):
        print(f"[Worker {worker_id}] Skipping {filename_stem} (Done)")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Worker {worker_id}] Error opening {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    half_width = width // 2

    # CSV Setup
    header = ['frame', 'timestamp', 'success'] + [f'AU_{i:02d}' for i in range(35)] 
    
    with open(left_csv_path, 'w', newline='') as f_l, open(right_csv_path, 'w', newline='') as f_r:
        writer_l = csv.writer(f_l)
        writer_r = csv.writer(f_r)
        writer_l.writerow(header)
        writer_r.writerow(header)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            timestamp = frame_idx / fps

            # --- LEFT ---
            frame_left = frame[0:height, 0:half_width]
            # out_left.write(frame_left) # VIDEO WRITING DISABLED FOR SPEED
            
            success_l, aus_l = process_frame(frame_left, face_detector, multitask_model, temp_handler)
            row_l = [frame_idx, timestamp, 1 if success_l else 0]
            row_l.extend(aus_l if success_l else [0.0]*35)
            writer_l.writerow(row_l)

            # --- RIGHT ---
            frame_right = frame[0:height, half_width:width]
            # out_right.write(frame_right) # VIDEO WRITING DISABLED FOR SPEED

            success_r, aus_r = process_frame(frame_right, face_detector, multitask_model, temp_handler)
            row_r = [frame_idx, timestamp, 1 if success_r else 0]
            row_r.extend(aus_r if success_r else [0.0]*35)
            writer_r.writerow(row_r)

            frame_idx += 1
            
            if frame_idx % 500 == 0:
                print(f"[Worker {worker_id}] {filename_stem}: Frame {frame_idx}", flush=True)

    cap.release()

# -----------------------------------------------------------------------------
# WORKER PROCESS
# -----------------------------------------------------------------------------
def worker_fn(rank, gpu_id, video_queue, args):
    """
    Worker process that initializes models on a specific GPU and consumes the queue.
    """
    try:
        # Assign this process to a specific GPU
        device = f"cuda:{gpu_id}"
        print(f"[Worker {rank}] Starting on {device}")
        
        # Load models locally (Crucial for multiprocessing)
        face_det, mtl_model = load_models(args.weights_root, device)
        
        # Create a temp handler for this process
        temp_handler = TempImageHandler()

        while not video_queue.empty():
            try:
                # Get next video from queue with a short timeout
                video_path = video_queue.get(timeout=2)
            except:
                break # Queue likely empty

            print(f"[Worker {rank}] Processing {video_path.name}")
            process_video(video_path, args.output_dir, face_det, mtl_model, temp_handler, rank)
        
        # Cleanup
        temp_handler.cleanup()
        print(f"[Worker {rank}] Finished.")

    except Exception as e:
        print(f"[Worker {rank}] CRASHED: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--weights_root', default='./external/OpenFace-3.0')
    parser.add_argument('--num_gpus', type=int, default=None, help="Manually set number of GPUs to use")
    args = parser.parse_args()

    # 1. Setup
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = list(input_path.glob("*.mp4"))
    print(f"Found {len(videos)} videos.")
    
    if not videos: return

    # 2. Detect GPUs
    if args.num_gpus:
        num_gpus = args.num_gpus
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs found. Exiting.")
            return

    print(f"Using {num_gpus} GPUs.")

    # 3. Fill Queue
    # We use a multiprocessing Manager Queue to share the list safely
    manager = mp.Manager()
    queue = manager.Queue()
    for v in videos:
        queue.put(v)

    # 4. Launch Workers
    # 'spawn' is required for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=worker_fn, args=(i, i, queue, args))
        p.start()
        processes.append(p)

    # 5. Wait for all to finish
    for p in processes:
        p.join()

    print("All workers finished.")

if __name__ == "__main__":
    main()
