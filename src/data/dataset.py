import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import scipy.io
import cv2
import os
import glob
import json
from src.model.summarizer import QueryEncoder
from src.preprocess.segmentation import get_segments


# We are moving away from on-the-fly extraction to pre-extracted features
# to speed up training on CPU.


class SumMeDataset(Dataset):
    """Dataset loader for SumMe (.npy frames + .mat annotations)."""
    
    def __init__(self, directory_path, feature_dir, query_encoder, device, max_frames=512):
        self.directory_path = directory_path
        self.feature_dir = feature_dir
        self.query_encoder = query_encoder
        self.max_frames = max_frames
        self.device = device
        
        # Load pre-encoded default query for SumMe
        self.query_embedding = torch.from_numpy(self.query_encoder.encode("Summarize this video")).float().squeeze(0)
        
        self.feature_files = glob.glob(os.path.join(feature_dir, "*.npy"))
        self.keys = [os.path.basename(f).replace('.npy', '') for f in self.feature_files]
        
        # Pre-calculate change points with caching
        self.change_points = {}
        cache_path = os.path.join(feature_dir, "summe_cps.json")
        
        if os.path.exists(cache_path):
            print("[SumMe] Loading change points from cache...")
            with open(cache_path, 'r') as f:
                self.change_points = json.load(f)
        else:
            print("[SumMe] Calculating change points (first time)...")
            for key in self.keys:
                feat = np.load(os.path.join(feature_dir, f"{key}.npy"))
                self.change_points[key] = get_segments(feat)
            with open(cache_path, 'w') as f:
                json.dump(self.change_points, f)
            
        print(f"[SumMe] Found {len(self.keys)} videos.")
            
    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, idx):
        video_name = self.keys[idx]
        
        # Load pre-extracted features: shape (N, 2048)
        features = np.load(os.path.join(self.feature_dir, f"{video_name}.npy"))
        
        # Load ground truth scores from .mat
        mat_path = os.path.join(self.directory_path, f"{video_name}.mat")
        mat_data = scipy.io.loadmat(mat_path)
        gt_score = mat_data['gt_score'].squeeze()
        
        n_gt = len(gt_score)
        n_feat = len(features)
        n_target = min(n_feat, n_gt, self.max_frames)
        
        # Sample features and scores to match
        # If video is shorter than max_frames, take all. If longer, sample evenly.
        if n_feat <= self.max_frames:
            feat_indices = np.arange(n_feat)
        else:
            feat_indices = np.linspace(0, n_feat - 1, self.max_frames, dtype=int)
            
        sampled_features = features[feat_indices]
        sampled_gt = gt_score[feat_indices]
        
        # Normalize gt_score to [0, 1]
        gt_max = sampled_gt.max()
        if gt_max > 0:
            sampled_gt = sampled_gt / gt_max
        
        cps = torch.tensor(self.change_points[video_name])
        return torch.from_numpy(sampled_features).float(), torch.from_numpy(sampled_gt).float(), self.query_embedding, cps


class TVSumDataset(Dataset):
    """Dataset loader for TVSum (raw .mp4 videos + TSV annotations)."""
    
    CATEGORY_MAP = {
        'VT': 'Changing Vehicle Tire',
        'VU': 'Getting Vehicle Unstuck',
        'GA': 'Grooming Animal',
        'MS': 'Making Sandwich',
        'PK': 'Parkour',
        'PR': 'Parade',
        'FM': 'Flash Mob',
        'BK': 'Bee Keeping',
        'BT': 'Bike Trick',
        'DS': 'Dog Show'
    }

    def __init__(self, data_dir, feature_dir, query_encoder, device, max_frames=512):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.query_encoder = query_encoder
        self.max_frames = max_frames
        self.device = device
        
        # Parse TSV annotations
        # Format: video_id \t category \t comma-separated scores
        # Multiple rows per video (one per annotator) — average them
        tsv_path = os.path.join(data_dir, "ydata-tvsum50-anno.tsv")
        self.annotations = {}  # video_id -> averaged scores
        
        raw_scores = {}
        with open(tsv_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                vid_id = parts[0]
                scores = np.array([int(s) for s in parts[2].split(',')])
                if vid_id not in raw_scores:
                    raw_scores[vid_id] = []
                raw_scores[vid_id].append(scores)
        
        # Average across annotators for each video
        for vid_id, score_list in raw_scores.items():
            self.annotations[vid_id] = np.mean(score_list, axis=0)
        
        self.keys = []
        self.query_embeddings = {}  # vid_id -> query vector
        self.change_points = {}
        
        # We also need the categories to build queries
        # Read info file for categories and titles
        info_path = os.path.join(data_dir, "ydata-tvsum50-info.tsv")
        vid_to_cat = {}
        with open(info_path, 'r') as f:
            next(f) # skip header
            for line in f:
                parts = line.strip().split('\t')
                vid_to_cat[parts[1]] = parts[0]

        print("[TVSum] Loading change points and queries...")
        cache_path = os.path.join(feature_dir, "tvsum_cps.json")
        cached_cps = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached_cps = json.load(f)

        for vid_id in self.annotations:
            feat_path = os.path.join(feature_dir, f"{vid_id}.npy")
            if os.path.exists(feat_path):
                self.keys.append(vid_id)
                # Encode query
                cat_desc = self.CATEGORY_MAP.get(vid_to_cat.get(vid_id, ""), "Summarize this video")
                emb = self.query_encoder.encode(cat_desc)
                self.query_embeddings[vid_id] = torch.from_numpy(emb).float().squeeze(0)
                
                # Change points
                if vid_id in cached_cps:
                    self.change_points[vid_id] = cached_cps[vid_id]
                else:
                    feat = np.load(feat_path)
                    self.change_points[vid_id] = get_segments(feat)
        
        if not os.path.exists(cache_path):
            with open(cache_path, 'w') as f:
                json.dump(self.change_points, f)
        
        print(f"[TVSum] Found {len(self.keys)} videos.")
            
    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, idx):
        vid_id = self.keys[idx]
        
        gt_score = self.annotations[vid_id]
        
        # Load pre-extracted features
        features = np.load(os.path.join(self.feature_dir, f"{vid_id}.npy"))
        
        n_gt = len(gt_score)
        n_feat = len(features)
        n_target = min(n_feat, n_gt, self.max_frames)
        
        # Sample to match target length
        if n_feat <= self.max_frames:
            feat_indices = np.arange(n_feat)
        else:
            feat_indices = np.linspace(0, n_feat - 1, self.max_frames, dtype=int)
            
        sampled_features = features[feat_indices]
        sampled_gt = gt_score[feat_indices]
        
        # Normalize
        gt_max = sampled_gt.max()
        if gt_max > 0:
            sampled_gt = sampled_gt / gt_max
            
        cps = torch.tensor(self.change_points[vid_id])
        return torch.from_numpy(sampled_features).float(), torch.from_numpy(sampled_gt).float(), self.query_embeddings[vid_id], cps


def get_combined_dataset(summe_dir, tvsum_data_dir, feature_dir, device=None, max_frames=512):
    """Create a combined dataset using pre-extracted features."""
    from torch.utils.data import ConcatDataset
    
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_encoder = QueryEncoder(device=device)
    
    summe_feat_dir = os.path.join(feature_dir, "summe")
    tvsum_feat_dir = os.path.join(feature_dir, "tvsum")
    
    summe_dataset = SumMeDataset(summe_dir, summe_feat_dir, query_encoder, device, max_frames)
    tvsum_dataset = TVSumDataset(tvsum_data_dir, tvsum_feat_dir, query_encoder, device, max_frames)
    
    combined = ConcatDataset([summe_dataset, tvsum_dataset])
    print(f"[Combined] Total videos: {len(combined)} (SumMe: {len(summe_dataset)}, TVSum: {len(tvsum_dataset)})")
    
    return combined
