import cv2
import numpy as np

class SummaryGenerator:
    def __init__(self, threshold=None, top_k=None):
        self.threshold = threshold
        self.top_k = top_k

    def generate_summary(self, frames, frame_indices, scores, output_path, fps=30,
                         segments=None, target_duration_secs=120, frames_per_sample=1):
        """
        Selects key shots and generates a summary video of the target duration.
        Each selected keyframe is held for its proportional time slice.

        target_duration_secs: desired output length (120=2min, 300=5min)
        frames_per_sample: how many original frames each sampled frame represents
        """
        n_frames = len(frames)
        assert n_frames == len(scores), "Number of frames must match number of scores"

        # How long to hold each keyframe: proportional to skip rate, capped at 2s
        hold_frames = min(int(frames_per_sample), int(fps * 2))
        hold_frames = max(hold_frames, 1)

        # Time-based capacity: how many sampled frames needed for target duration
        target_original_frames = int(target_duration_secs * fps)
        capacity = max(int(target_original_frames / hold_frames), 1)
        capacity = min(capacity, n_frames)
        print(f"Target: {target_duration_secs}s | hold={hold_frames}f | capacity={capacity} sampled frames")

        if segments is None:
            selected_indices = sorted(np.argsort(scores)[-capacity:])
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
                    start, end = segments[i-1]
                    selected_indices.extend(range(start, min(end, n_frames)))
                    w -= weights[i-1]

            selected_indices.sort()

        if not selected_indices:
            print("Warning: Knapsack selected empty. Falling back to top frames.")
            selected_indices = sorted(np.argsort(scores)[-max(capacity, 1):])

        # Write output video — repeat each frame for hold_frames to fill duration
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        estimated_secs = len(selected_indices) * hold_frames / fps
        print(f"Generating summary: {len(selected_indices)} keyframes x {hold_frames} holds = ~{estimated_secs:.0f}s output")

        for idx in selected_indices:
            frame_rgb = frames[idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            for _ in range(hold_frames):
                out.write(frame_bgr)

        out.release()
        return output_path
