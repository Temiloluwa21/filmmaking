import os
import torch
import numpy as np
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuration
SUMME_DIR = r"C:\Users\Temiloluwa\Downloads\archive"
TVSUM_VIDEO_DIR = r"C:\Users\Temiloluwa\Downloads\archive (1)\tvsum_dataset\ydata-tvsum50-video\video"
OUTPUT_DIR = "clip_features"
MAX_FRAMES = 512 

def extract_clip_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize CLIP model via sentence-transformers (more robust loading)
    print("Loading CLIP (clip-ViT-B-32)...")
    model = SentenceTransformer('clip-ViT-B-32', device=device)
    model.eval()

    os.makedirs(os.path.join(OUTPUT_DIR, "summe"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "tvsum"), exist_ok=True)

    # --- SumMe Extraction ---
    print("\n--- Processing SumMe with CLIP ---")
    summe_files = glob.glob(os.path.join(SUMME_DIR, "*.npy"))
    for f in tqdm(summe_files):
        video_name = os.path.basename(f).replace(".npy", "")
        if os.path.exists(os.path.join(OUTPUT_DIR, "summe", f"{video_name}.npy")):
            continue
            
        raw_frames = np.load(f) # [N, 128, 128, 3]
        n_raw = len(raw_frames)
        indices = np.linspace(0, n_raw - 1, min(n_raw, MAX_FRAMES), dtype=int)
        sampled_frames = raw_frames[indices]

        # Extract using sentence-transformers (automatically handles batches and normalization)
        features = model.encode(
            list(sampled_frames), 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        np.save(os.path.join(OUTPUT_DIR, "summe", f"{video_name}.npy"), features)

    # --- TVSum Extraction ---
    print("\n--- Processing TVSum with CLIP ---")
    import cv2
    video_files = glob.glob(os.path.join(TVSUM_VIDEO_DIR, "*.mp4"))
    for f in tqdm(video_files):
        vid_id = os.path.basename(f).replace(".mp4", "")
        if os.path.exists(os.path.join(OUTPUT_DIR, "tvsum", f"{vid_id}.npy")):
            continue
            
        # Manually extract frames for TVSum to avoid VideoProcessor overhead
        cap = cv2.VideoCapture(f)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: continue
        
        indices = np.linspace(0, total_frames - 1, min(total_frames, MAX_FRAMES), dtype=int)
        sampled_frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame_rgb)
        cap.release()
        
        if not sampled_frames: continue
        
        features = model.encode(
            sampled_frames, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        np.save(os.path.join(OUTPUT_DIR, "tvsum", f"{vid_id}.npy"), features)

if __name__ == "__main__":
    try:
        extract_clip_features()
    except Exception as e:
        print(f"Extraction failed: {e}")
