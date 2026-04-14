import cv2
import numpy as np
import os
from moviepy import VideoFileClip, concatenate_videoclips

class SummaryGenerator:
    def __init__(self, threshold=None, top_k=None):
        self.threshold = threshold
        self.top_k = top_k

    def generate_summary(self, video_path, frames, frame_indices, scores, output_path, fps=30,
                         segments=None, target_duration_secs=120, frames_per_sample=1):
        """
        Selects key shots based on scores and generates a summary video with original audio.
        Uses moviepy to stitch actual sub-clips from the original video for maximum quality.
        """
        n_frames = len(frames)
        assert n_frames == len(scores), "Number of frames must match number of scores"

        # Time-based capacity: how many total sampled frames can we afford?
        # Note: hold_frames isn't used for moviepy since we take actual clips, 
        # but we use the same capacity logic to decide how many segments to pick.
        target_original_frames = int(target_duration_secs * fps)
        capacity = max(target_original_frames // int(frames_per_sample or 1), 1)
        capacity = min(capacity, n_frames)

        print(f"Target duration: {target_duration_secs}s | Selection Capacity: {capacity} frames")

        selected_segments = []
        if segments is None:
            # Fallback to simple top frames if no segments
            selected_indices = sorted(np.argsort(scores)[-capacity:])
            # For moviepy, we need continuous clips. We'll group them for smoother audio.
            if selected_indices:
                curr_start = selected_indices[0]
                for i in range(1, len(selected_indices)):
                    if selected_indices[i] != selected_indices[i-1] + 1:
                        selected_segments.append((curr_start, selected_indices[i-1] + 1))
                        curr_start = selected_indices[i]
                selected_segments.append((curr_start, selected_indices[-1] + 1))
        else:
            # SHOT-BASED SELECTION via DP Knapsack
            n_segs = len(segments)
            seg_scores = []
            for i in range(n_segs):
                start, end = segments[i]
                if start < n_frames:
                    seg_scores.append(np.mean(scores[start:min(end, n_frames)]))
                else:
                    seg_scores.append(0)

            weights = [min(segments[i][1], n_frames) - segments[i][0] for i in range(n_segs)]
            values = [int(s * 1000) for s in seg_scores]

            n = len(values)
            dp = np.zeros((n + 1, capacity + 1))
            for i in range(1, n + 1):
                for w in range(1, capacity + 1):
                    if weights[i-1] <= w:
                        dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                    else:
                        dp[i][w] = dp[i-1][w]

            selected_indices = []
            w = capacity
            for i in range(n, 0, -1):
                if dp[i][w] != dp[i-1][w]:
                    selected_segments.append(segments[i-1])
                    w -= weights[i-1]
            
            selected_segments.sort()

        if not selected_segments:
            print("Warning: Selection failed. Falling back to simple summary.")
            return None

        # --- High Quality Stitching with Audio ---
        try:
            print(f"Stitching {len(selected_segments)} clips from original video with audio...")
            video = VideoFileClip(video_path)
            clips = []
            
            for start_idx, end_idx in selected_segments:
                # Map sampled indices back to original timestamps
                # frame_indices[i] is the original frame number
                # time = original_frame_number / original_fps
                t_start = frame_indices[start_idx] / fps
                t_end = frame_indices[min(end_idx-1, len(frame_indices)-1)] / fps
                
                # Small safety buffer to ensure valid clip duration
                if t_end > t_start:
                    clips.append(video.subclipped(t_start, t_end))

            if not clips:
                print("Error: No valid clips extracted.")
                video.close()
                return None

            final_summary = concatenate_videoclips(clips)
            
            # Write with professional encoding (H.264/AAC) for browser compatibility
            final_summary.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True,
                logger=None # Suppress verbose progress bars
            )
            
            # Cleanup
            for c in clips: c.close()
            final_summary.close()
            video.close()
            
            return output_path

        except Exception as e:
            print(f"Error during audio-synced generation: {e}")
            import traceback
            traceback.print_exc()
            return None
