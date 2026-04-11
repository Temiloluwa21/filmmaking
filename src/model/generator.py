import cv2
import numpy as np

class SummaryGenerator:
    def __init__(self, threshold=None, top_k=None):
        self.threshold = threshold
        self.top_k = top_k

    def generate_summary(self, frames, frame_indices, scores, output_path, fps=30, segments=None):
        """
        Selects key shots based on scores and generates a coherent summary video.
        
        frames: list of numpy arrays (RGB images)
        frame_indices: the original frame index for each frame
        scores: array of importance scores from the model
        output_path: where to save the generated video
        fps: playback speed
        segments: [[start, end], ...] shot boundaries
        """
        n_frames = len(frames)
        assert n_frames == len(scores), "Number of frames must match number of scores"
        
        if segments is None:
            # Fallback to simple top-k if segments aren't provided
            k = int(n_frames * 0.15)
            selected_indices = sorted(np.argsort(scores)[-k:])
        else:
            # SHOT-BASED SELECTION (Industry Standard)
            n_segs = len(segments)
            seg_scores = []
            for i in range(n_segs):
                start, end = segments[i]
                if start < n_frames:
                    seg_scores.append(np.mean(scores[start:min(end, n_frames)]))
                else:
                    seg_scores.append(0)
            
            # Use DP Knapsack for optimal selection (15% budget)
            capacity = int(n_frames * 0.15)
            weights = [min(segments[i][1], n_frames) - segments[i][0] for i in range(n_segs)]
            values = [int(s * 1000) for s in seg_scores]
            
            # DP Table
            n = len(values)
            dp = np.zeros((n + 1, capacity + 1))
            for i in range(1, n + 1):
                for w in range(1, capacity + 1):
                    if weights[i-1] <= w:
                        dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                    else:
                        dp[i][w] = dp[i-1][w]
            
            # Backtrack
            selected_indices = []
            w = capacity
            for i in range(n, 0, -1):
                if dp[i][w] != dp[i-1][w]:
                    start, end = segments[i-1]
                    selected_indices.extend(range(start, min(end, n_frames)))
                    w -= weights[i-1]
            
            selected_indices.sort()

        if not selected_indices:
            print("No frames selected. Returning empty.")
            return

        # Prepare to write video
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Generating summary with {len(selected_indices)} frames.")
        
        for idx in selected_indices:
            frame_rgb = frames[idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        return output_path
