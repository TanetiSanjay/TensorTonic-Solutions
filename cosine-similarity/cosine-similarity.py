import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.asarray(a, dtype = np.float32), np.asarray(b, dtype = np.float32)

    cosine = (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-16)

    print(cosine.shape)
    
    return float(cosine)