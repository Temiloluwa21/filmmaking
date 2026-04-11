import numpy as np

class Evaluator:
    @staticmethod
    def calculate_metrics(predicted_indices, ground_truth_indices):
        """Standard frame-based metrics."""
        S = set(predicted_indices)
        G = set(ground_truth_indices)
        overlap = S.intersection(G)
        precision = len(overlap) / len(S) if len(S) > 0 else 0.0
        recall = len(overlap) / len(G) if len(G) > 0 else 0.0
        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f_score": f_score}

    @staticmethod
    def evaluate_summary(predicted_scores, user_summary, cps, n_frames):
        """
        Shot-based evaluation (Standard for TVSum/SumMe).
        predicted_scores: [n_frames]
        user_summary: [n_frames] (binary ground truth)
        cps: [n_segments, 2] start/end of shots
        n_frames: total frames
        """
        # 1. Compute shot scores (average of frame scores in shot)
        n_segs = len(cps)
        seg_scores = []
        for i in range(n_segs):
            start, end = cps[i]
            if start < n_frames:
                seg_scores.append(np.mean(predicted_scores[start:min(end, n_frames)]))
            else:
                seg_scores.append(0)
        
        # 2. Select shots using True 0/1 Knapsack (DP)
        capacity = int(n_frames * 0.15)
        weights = [min(cps[i][1], n_frames) - cps[i][0] for i in range(n_segs)]
        values = [int(s * 1000) for s in seg_scores] # Scale scores to integers for DP
        
        # Standard DP Knapsack
        n = len(values)
        dp = np.zeros((n + 1, capacity + 1))
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Backtrack to find selected shots
        summary_indices = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                summary_indices.append(i-1)
                w -= weights[i-1]
        
        # 3. Create binary prediction mask
        pred_summary = np.zeros(n_frames, dtype=int)
        for idx in summary_indices:
            start, end = cps[idx]
            pred_summary[start:min(end, n_frames)] = 1
            
        # 4. Calculate Overlap
        overlap = np.sum(pred_summary * user_summary)
        precision = overlap / np.sum(pred_summary) if np.sum(pred_summary) > 0 else 0
        recall = overlap / np.sum(user_summary) if np.sum(user_summary) > 0 else 0
        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "f_score": f_score,
            "precision": precision,
            "recall": recall,
            "accuracy": np.mean(pred_summary == user_summary)
        }
