import numpy as np

def relu(x):
    return np.where(x<=0, 0, x)

def identity_block(x, W1, W2):
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    
    return (
        relu(
            relu(x @ W1.T) @ W2.T + x
        )
    )