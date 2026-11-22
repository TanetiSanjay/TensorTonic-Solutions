import numpy as np

def pca_projection(X: list, k: int) -> list:

    x = np.asarray(X, dtype=np.float32)

    xc = x - x.mean(axis=0)

    c = xc.T @ xc / (x.shape[0] - 1)

    u, s, vh = np.linalg.svd(c)

    return xc @ vh[:k, :].T
    
    

    