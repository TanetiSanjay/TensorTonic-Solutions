import numpy as np

def dot_product(x: list, y: list) -> float:
    return float(np.array(x, dtype = np.float32).T @ np.array(y, np.float32))