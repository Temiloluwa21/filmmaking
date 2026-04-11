import os
import torch
import numpy as np
import glob
import scipy.io
from tqdm import tqdm
from src.preprocess.video_processor import VideoProcessor

# Configuration
SUMME_DIR = r"C:\Users\Temiloluwa\Downloads\archive"
TVSUM_DATA_DIR = r"C:\Users\Temiloluwa\Downloads\archive (1)\tvsum_dataset\ydata-tvsum50-data\data"
TVSUM_VIDEO_DIR = r"C:\Users\Temiloluwa\Downloads\archive (1)\tvsum_dataset\ydata-tvsum50-video\video"

OUTPUT_DIR = "extracted_features"
MAX_FRAMES = 100 # Consistent with training settings

def pre_extract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VideoProcessor(device=device)
    
    os.makedirs(os.path.join(OUTPUT_DIR, "summe"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "tvsum"), exist_ok=True)

    # --- SumMe Extraction ---
    print("\n--- Processing SumMe ---")
    summe_files = glob.glob(os.path.join(SUMME_DIR, "*.npy"))
    for f in tqdm(summe_files):
        video_name = os.path.basename(f).replace(".npy", "")
        # SumMe .npy files are raw frames [N, 128, 128, 3]
        raw_frames = np.load(f)
        n_raw = len(raw_frames)
        
        # Sample frames uniformly (matching dataset.py logic)
        indices = np.linspace(0, n_raw - 1, min(n_raw, MAX_FRAMES), dtype=int)
        sampled_frames = raw_frames[indices]
        
        # Extract features
        features = processor.extract_features(sampled_frames)
        
        # Save features
        np.save(os.path.join(OUTPUT_DIR, "summe", f"{video_name}.npy"), features)

    # --- TVSum Extraction ---
    print("\n--- Processing TVSum ---")
    # Get video IDs from the info file (or just the directory)
    video_files = glob.glob(os.path.join(TVSUM_VIDEO_DIR, "*.mp4"))
    for f in tqdm(video_files):
        vid_id = os.path.basename(f).replace(".mp4", "")
        
        # Use VideoProcessor to extract frames and features
        # We'll sample min(total_frames, MAX_FRAMES)
        frames, _, _ = processor.extract_frames(f)
        n_frames = len(frames)
        
        if n_frames > MAX_FRAMES:
            indices = np.linspace(0, n_frames - 1, MAX_FRAMES, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames
            
        features = processor.extract_features(sampled_frames)
        
        # Save features
        np.save(os.path.join(OUTPUT_DIR, "tvsum", f"{vid_id}.npy"), features)

if __name__ == "__main__":
    pre_extract()
