import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats.mstats import zscore


def dtw(x, y, dist, l=1, warp=1, z_normalize=False):

    if z_normalize:
        x = zscore(x)
        y = zscore(y)

    series_len = len(x)
    distance_cost = np.full((series_len + 1, series_len + 1), np.inf)
    distance_cost[0, 0] = 0
    ident = int(l * series_len)

    pairs = distance_cost[1:, 1:]
    for i in range(series_len):
        for j in range(max(0, i - ident), min(series_len, i + ident + 1)):
            pairs[i, j] = dist(x[i], y[j])

    pairwise_distances = pairs.copy()
    for i in range(1, series_len + 1):
        for j in range(max(1, i - ident), min(series_len + 1, i + ident + 1)):
            min_list = []
            for k in range(1, warp + 1):
                i_k = max(i - k, 0)
                j_k = max(j - k, 0)
                min_list += [distance_cost[i_k, j], distance_cost[i, j_k], distance_cost[i_k, j_k]]
            distance_cost[i, j] += min(min_list)
    
    path, path_cost = _traceback(distance_cost)
            
    return path_cost, path, distance_cost[1:, 1:], pairwise_distances


def _traceback(D):
    i = D.shape[0] - 1
    j = D.shape[1] - 1
    p, q = [], []
    while (i > 0) or (j > 0):
        p.insert(0, i)
        q.insert(0, j)
        tb = np.argmin((D[i - 1, j - 1], D[i, j - 1], D[i - 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            j -= 1
        else:
            i -= 1
            
    distance = D[-1, -1]
    
    return (np.array(p), np.array(q)), distance / len(p)