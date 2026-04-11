import numpy as np
import os
import json

def cpd_nonhomogeneous(K, n_cp, wnorm=None):
    """Optimized non-homogeneous change point detection."""
    n = K.shape[0]
    if wnorm is None:
        wnorm = np.ones(n)
        
    diag_K = np.diag(K)
    diag_cumsum = np.concatenate(([0], np.cumsum(diag_K)))
    
    # Precompute K_sum[i, j] = sum(K[i:j+1, i:j+1])
    # This is the most expensive part pre-DP
    K_cumsum = np.cumsum(np.cumsum(K, axis=0), axis=1)

    def get_sum(i, j):
        res = K_cumsum[j, j]
        if i > 0:
            res -= K_cumsum[i-1, j]
            res -= K_cumsum[j, i-1]
            res += K_cumsum[i-1, i-1]
        return res

    # dp[m, i] table
    dp = np.full((n_cp + 1, n), float('inf'))
    backtrack = np.zeros((n_cp + 1, n), dtype=int)

    # Base case: 1 segment
    for i in range(n):
        sum_kernel = get_sum(0, i)
        sum_diag = diag_cumsum[i+1] - diag_cumsum[0]
        dp[0, i] = sum_diag - (sum_kernel / (i + 1))

    # DP iterations
    for m in range(1, n_cp + 1):
        # We can optimize the search for j if we assume a minimum segment length
        # but for now let's just use the loop
        for i in range(m, n):
            # We can vectorize the inner loop
            prev_costs = dp[m-1, m-1:i]
            # Calculating get_cost(j+1, i) for all j in range(m-1, i)
            # j_plus_1 range: m to i
            # sizes range: i-m+1 down to 1
            j_indices = np.arange(m-1, i)
            j_plus_1 = j_indices + 1
            sizes = i - j_plus_1 + 1
            
            # get_sum(j+1, i) vectorized
            sum_kernels = K_cumsum[i, i] - K_cumsum[j_indices, i] - K_cumsum[i, j_indices] + K_cumsum[j_indices, j_indices]
            sum_diags = diag_cumsum[i+1] - diag_cumsum[j_plus_1]
            costs = sum_diags - (sum_kernels / sizes)
            
            total_costs = prev_costs + costs
            best_idx = np.argmin(total_costs)
            dp[m, i] = total_costs[best_idx]
            backtrack[m, i] = m - 1 + best_idx

    cps = []
    curr_i = n - 1
    for m in range(n_cp, 0, -1):
        curr_i = backtrack[m, curr_i]
        cps.append(curr_i + 1)
        
    return sorted(cps)

def get_segments(features, n_cp_max=20, subsample=5):
    """
    Subsampled KTS for speed. 
    Standard summe/tvsum features are 2fps, but original videos might be 25-30fps.
    If features is already low-fps, subsample=1.
    """
    n_orig = features.shape[0]
    
    # Subsample if video is long
    if n_orig > 250:
        idx = np.arange(0, n_orig, subsample)
        features_sub = features[idx]
    else:
        features_sub = features
        subsample = 1
        
    n = features_sub.shape[0]
    if n <= 1:
        return [[0, n_orig]]
        
    K = np.dot(features_sub, features_sub.T)
    cps_sub = cpd_nonhomogeneous(K, min(n_cp_max, n-1))
    
    # Map back to original indices
    cps = [cp * subsample for cp in cps_sub]
    
    boundaries = [0] + cps + [n_orig]
    segments = []
    for i in range(len(boundaries) - 1):
        segments.append([int(boundaries[i]), int(boundaries[i+1])])
        
    return segments
