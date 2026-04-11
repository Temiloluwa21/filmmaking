from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import sys

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
try:
    processor = VideoProcessor(target_fps=2)
    device = processor.device
    
    query_encoder = QueryEncoder(device=device)
    summarizer = VideoSummarizer().to(device)
    
    weights_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_weights_best.pth')
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        summarizer.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Warning: model_weights_best.pth not found. Model will use random initialization.")
    
    summarizer.eval()
    print("Models ready.")
except Exception as e:
    print(f"Error initializing models during startup: {e}")

@app.post("/api/summarize")
async def summarize_video(
    video: UploadFile = File(...), 
    query: str = Form(...)
):
    if not video.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
        
    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}.mp4")
    out_path = os.path.join(OUTPUT_DIR, f"{task_id}_summary.mp4")
    
    # Save upload
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    try:
        # Step 1: Preprocess (Extract frames, features, and segments)
        features, frames, frame_indices, fps, segments = processor.process_video(video_path)
        
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
        generator.generate_summary(frames, frame_indices, scores, out_path, fps=fps, segments=segments)

        return FileResponse(out_path, media_type="video/mp4", filename=f"summary_{task_id}.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{task_id}")
async def download_summary(task_id: str):
    file_path = os.path.join(OUTPUT_DIR, f"{task_id}_summary.mp4")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=f"summary_{task_id}.mp4")
    raise HTTPException(status_code=404, detail="File not found")
