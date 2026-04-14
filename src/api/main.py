from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import sys
import traceback
import time

# Update path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocess.video_processor import VideoProcessor
from src.model.summarizer import VideoSummarizer, QueryEncoder
from src.model.generator import SummaryGenerator
import torch
import numpy as np

app = FastAPI(title="AI Video Summarization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global task tracker
TASK_STATUS = {}

# Initialize Models
print("Initializing models...")
processor = None
query_encoder = None
summarizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    processor = VideoProcessor(target_fps=2)
    device = processor.device
    query_encoder = QueryEncoder(device=device)
    summarizer = VideoSummarizer().to(device)
    
    weights_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_weights_best.pth')
    if os.path.exists(weights_path):
        summarizer.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    summarizer.eval()
    print("Models ready.")
except Exception as e:
    print(f"Error initializing models: {e}")

def run_summarization_task(task_id: str, video_path: str, out_path: str, query: str):
    """Background worker for progress tracking and cleanup."""
    try:
        TASK_STATUS[task_id] = {"progress": 10, "status": "Reading Video Feed..."}
        
        # Run preprocessing
        TASK_STATUS[task_id] = {"progress": 25, "status": "Reading video frames..."}
        features, frames, frame_indices, fps, segments, skip_frames = processor.process_video(video_path)
        
        # Convert search text to vector
        TASK_STATUS[task_id] = {"progress": 45, "status": "Processing search query..."}
        q_embed = query_encoder.encode(query) # MiniLM for the LSTM model
        q_clip = processor.get_text_features(query) # CLIP for direct visual similarity
        
        # Run the model
        TASK_STATUS[task_id] = {"progress": 65, "status": "Processing sequences..."}
        features_tensor = torch.tensor(features).unsqueeze(0).to(device)
        q_tensor = torch.tensor(q_embed).to(device)
        with torch.no_grad():
            model_scores = summarizer(features_tensor, q_tensor)
            model_scores = model_scores.squeeze(0).cpu().numpy()
            
        # Blend the search scores with the model output
        if q_clip is not None:
            # Calculate cosine similarity between each frame (features) and query (q_clip)
            # Both are already normalized to unit length
            clip_sims = np.dot(features, q_clip.T).squeeze()
            # Rescale similarity from roughly [-0.2, 0.4] to [0, 1] for blending
            clip_sims = np.clip((clip_sims + 0.1) / 0.5, 0, 1)
            # Blend: 40% Generic Importance, 60% Explicit Prompt Alignment
            scores = (0.4 * model_scores) + (0.6 * clip_sims)
            print(f"Blended semantic scores: sim_avg={np.mean(clip_sims):.2f}")
        else:
            scores = model_scores
            
        # Cut and stitch the video based on scores
        TASK_STATUS[task_id] = {"progress": 85, "status": "Saving video highlights..."}
        
        # New Rule: Duration / 6, Minimum 2m (120s)
        duration_secs = (frame_indices[-1] / fps) if frame_indices else 60
        target_secs = max(120, int(duration_secs / 6))
        print(f"Scaling Summary: {duration_secs:.0f}s source -> {target_secs}s summary (1/6th ration)")

        generator = SummaryGenerator()
        result = generator.generate_summary(
            video_path, frames, frame_indices, scores, out_path, 
            fps=fps, segments=segments, 
            target_duration_secs=target_secs, 
            frames_per_sample=skip_frames
        )

        if result:
            TASK_STATUS[task_id] = {"progress": 100, "status": "Ready", "file_id": task_id}
            # Clean up original file
            if os.path.exists(video_path):
                os.remove(video_path)
        else:
            TASK_STATUS[task_id] = {"progress": 0, "status": "Error: Generation Failed"}

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in task {task_id}: {error_msg}")
        TASK_STATUS[task_id] = {"progress": 0, "status": f"Error: {str(e)}"}
        with open("server_error.log", "a") as f:
            f.write(f"\n--- {time.ctime()} ---\n{error_msg}\n")

@app.post("/api/summarize")
async def summarize_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...), 
    query: str = Form("")
):
    if processor is None or summarizer is None:
        raise HTTPException(status_code=503, detail="Models not initialized.")
    if not video.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 supported.")
        
    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}.mp4")
    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_summary.mp4")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    # Start background task
    background_tasks.add_task(run_summarization_task, task_id, video_path, out_path, query)
    
    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    return TASK_STATUS.get(task_id, {"progress": 0, "status": "Waiting..."})

@app.get("/api/download/{task_id}")
async def download_summary(task_id: str):
    file_path = os.path.join(OUTPUT_DIR, f"{task_id}_summary.mp4")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=f"summary_{task_id}.mp4")
    raise HTTPException(status_code=404, detail="File not found")

# Serve Frontend
frontend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print("Warning: Frontend directory not found.")
