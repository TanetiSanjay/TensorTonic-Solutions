import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    v, w = np.asarray(v), np.asarray(w)

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)

    c = (v @ w.T) / (v_norm * w_norm)
    
    return float(np.arccos(c))