import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class VideoProcessor:
    def __init__(self, target_fps=2, device=None):
        """
        Setup the video processor and load the model.
        """
        self.target_fps = target_fps
        self.MAX_FRAMES = 300  # Hard cap: guarantees <30s processing on CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model - using local files if available to avoid download issues
        print(f"Loading CLIP on {self.device}...")
        self.model_name = "openai/clip-vit-base-patch32"
        try:
            self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            # Download if not in cache
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

    def extract_frames(self, video_path):
        """
        Extract frames from the video at a set resolution.
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

        frames_480p = []   # 480p stored for high-quality output
        frame_indices = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % skip_frames == 0:
                # Store at 480p (good output quality, manageable memory)
                h, w = frame.shape[:2]
                new_w = int(w * 480 / h) if h > 0 else 854
                frame_480 = cv2.resize(frame, (new_w, 480))
                rgb_frame = cv2.cvtColor(frame_480, cv2.COLOR_BGR2RGB)
                frames_480p.append(rgb_frame)
                frame_indices.append(count)
                if len(frames_480p) >= self.MAX_FRAMES:
                    break
            count += 1

        cap.release()
        return frames_480p, frame_indices, orig_fps, skip_frames

    def extract_features(self, frames_480p):
        """
        Run the model on extracted frames to get features.
        """
        all_features = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(frames_480p), batch_size):
                batch = frames_480p[i : i + batch_size]
                # Downscale ONLY for CLIP — 160px is enough for semantic understanding
                small_batch = [cv2.resize(f, (160, 160)) for f in batch]
                inputs = self.processor(images=small_batch, return_tensors="pt", padding=True).to(self.device)
                feat = self.model.get_image_features(**inputs)
                
                if not isinstance(feat, torch.Tensor):
                    feat = getattr(feat, 'image_embeds', getattr(feat, 'pooler_output', feat[0]))

                feat = feat / feat.norm(dim=-1, keepdim=True)
                all_features.append(feat.cpu().numpy())
                
        return np.concatenate(all_features, axis=0)

    def get_text_features(self, text):
        """
        Get features for the text query.
        """
        if not text:
            return None
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(text_features, 'text_embeds', getattr(text_features, 'pooler_output', text_features[0]))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def process_video(self, video_path):
        """
        Full pipeline: Extract 480p frames -> CLIP features -> KTS segments.
        Returns skip_frames so generator can compute output duration correctly.
        """
        from src.preprocess.segmentation import get_segments
        
        frames, frame_indices, fps, skip_frames = self.extract_frames(video_path)
        features = self.extract_features(frames)
        segments = get_segments(features)
        
        return features, frames, frame_indices, fps, segments, skip_frames
