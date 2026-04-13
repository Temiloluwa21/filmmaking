import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class VideoProcessor:
    def __init__(self, target_fps=2, frame_size=(160, 160), device=None):
        """
        Initializes the Video Processor for CLIP semantic feature encoding.
        """
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.MAX_FRAMES = 300  # Hard cap: guarantees <30s processing on CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP ViT-B/32 — offline-first to avoid network failures
        print(f"Loading CLIP on {self.device}...")
        self.model_name = "openai/clip-vit-base-patch32"
        try:
            self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            print("Cache miss — downloading CLIP from HuggingFace...")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def extract_frames(self, video_path):
        """
        Extracts frames from a video with adaptive FPS to cap total frames at MAX_FRAMES.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_secs = total_video_frames / orig_fps

        # Adaptive FPS: scale down sampling for long videos to stay within MAX_FRAMES
        ideal_sample_count = min(self.MAX_FRAMES, max(50, int(duration_secs * self.target_fps)))
        skip_frames = max(1, int(total_video_frames / ideal_sample_count))
        print(f"Video: {duration_secs:.1f}s | Sampling every {skip_frames} frames (~{ideal_sample_count} total)")

        frames = []
        frame_indices = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % skip_frames == 0:
                # Resize immediately on read to cut memory usage
                frame_small = cv2.resize(frame, self.frame_size)
                rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                frame_indices.append(count)
                if len(frames) >= self.MAX_FRAMES:
                    break

            count += 1

        cap.release()
        return frames, frame_indices, orig_fps

    def extract_features(self, frames):
        """
        Extracts semantic features using CLIP with speed-optimized batching.
        """
        all_features = []
        batch_size = 64  # Doubled from 32 for faster throughput
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                inputs = self.processor(images=list(batch), return_tensors="pt", padding=True).to(self.device)
                feat = self.model.get_image_features(**inputs)
                
                # Handle HuggingFace version API discrepancies (Tuple/Object vs Tensor)
                if not isinstance(feat, torch.Tensor):
                    feat = getattr(feat, 'image_embeds', getattr(feat, 'pooler_output', feat[0]))

                # Normalize features
                feat = feat / feat.norm(dim=-1, keepdim=True)
                all_features.append(feat.cpu().numpy())
                
        return np.concatenate(all_features, axis=0)

    def process_video(self, video_path):
        """
        Runs the full preprocessing pipeline: Extraction -> Resizing/Normalization -> Feature Encoding -> Segmentation.
        Returns:
            features: np array of shape (N, 2048)
            frames: list of numpy arrays (the raw frames for later generation)
            frame_indices: list of original frame indices
            fps: original video fps
            segments: KTS shot boundaries [[start, end], ...]
        """
        from src.preprocess.segmentation import get_segments
        
        frames, frame_indices, fps = self.extract_frames(video_path)
        features = self.extract_features(frames)
        
        # Calculate segments (KTS) based on extracted features
        segments = get_segments(features)
        
        return features, frames, frame_indices, fps, segments
