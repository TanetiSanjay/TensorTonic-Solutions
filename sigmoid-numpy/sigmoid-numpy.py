import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    if isinstance(x, float):
        return 1 / (1 + np.exp(-x))
    else:
        x = np.asarray(x)
        return 1 / (1 + np.exp(-x))