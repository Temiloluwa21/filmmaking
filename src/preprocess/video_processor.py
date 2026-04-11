import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class VideoProcessor:
    def __init__(self, target_fps=2, frame_size=(224, 224), device=None):
        """
        Initializes the Video Processor for CLIP semantic feature encoding.
        """
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP ViT-B/32
        print(f"Loading CLIP on {self.device}...")
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def extract_frames(self, video_path):
        """
        Extracts frames from a video at a fixed frame rate.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps == 0:
            orig_fps = 30 # Default if unable to read

        # Calculate frame skip to achieve target fps
        skip_frames = max(int(orig_fps / self.target_fps), 1)

        frames = []
        frame_indices = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % skip_frames == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                frame_indices.append(count)

            count += 1

        cap.release()
        return frames, frame_indices, orig_fps

    def extract_features(self, frames):
        """
        Extracts semantic features from a list of frames using CLIP.
        """
        all_features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                inputs = self.processor(images=list(batch), return_tensors="pt", padding=True).to(self.device)
                feat = self.model.get_image_features(**inputs)
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
