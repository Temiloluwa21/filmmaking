from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import sys
import traceback

# Update path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocess.video_processor import VideoProcessor
from src.model.summarizer import VideoSummarizer, QueryEncoder
from src.model.generator import SummaryGenerator
import torch

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

# Initialize Models (Lazy loading or global is fine for prototype)
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
        print(f"Loading weights from {weights_path}...")
        summarizer.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else:
        print("Warning: model_weights_best.pth not found. Model will use random initialization.")
    
    summarizer.eval()
    print("Models ready.")
except Exception as e:
    print(f"Error initializing models during startup: {e}")

@app.post("/api/summarize")
async def summarize_video(
    video: UploadFile = File(...), 
    query: str = Form("")
):
    budget = 0.15
    if processor is None or summarizer is None:
        raise HTTPException(status_code=503, detail="AI Models failed to download/initialize. Please check your internet connection and restart the server.")
        
    if not video.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
        
    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}.mp4")
    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_summary.mp4")
    
    # Save upload
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    try:
        # Step 1: Preprocess (Extract 480p frames, features, segments)
        features, frames, frame_indices, fps, segments, skip_frames = processor.process_video(video_path)

        # Determine video duration and set output target
        duration_secs = (frame_indices[-1] / fps) if frame_indices else 60
        target_secs = 300 if duration_secs >= 600 else 120  # 5min for long, 2min for short
        print(f"Source duration: {duration_secs:.0f}s | Output target: {target_secs}s")
        
        # Step 2: Encode Query
        print(f"Encoding query: '{query}'")
        q_embed = query_encoder.encode(query)
        
        # Step 3: Run Model
        print("Running summarization model...")
        features_tensor = torch.tensor(features).unsqueeze(0).to(device)
        q_tensor = torch.tensor(q_embed).to(device)
        
        with torch.no_grad():
            scores = summarizer(features_tensor, q_tensor)
            scores = scores.squeeze(0).cpu().numpy()
            
        # Step 4: Generate Summary (Shot-Based)
        print("Generating summary video...")
        generator = SummaryGenerator()
        generator.generate_summary(frames, frame_indices, scores, out_path, fps=fps,
                                   segments=segments, target_duration_secs=target_secs,
                                   frames_per_sample=skip_frames)

        return FileResponse(out_path, media_type="video/mp4", filename=f"summary_{task_id}.mp4")

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in summarize: {error_msg}")
        with open("server_error.log", "w") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{task_id}")
async def download_summary(task_id: str):
    file_path = os.path.join(OUTPUT_DIR, f"{task_id}_summary.mp4")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=f"summary_{task_id}.mp4")
    raise HTTPException(status_code=404, detail="File not found")

# Serve Frontend statically at the root level (Must be mounted LAST to avoid catching /api routes)
frontend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print("Warning: Frontend directory not found. UI will not be served.")
