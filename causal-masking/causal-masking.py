import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:

    scores    = np.asarray(scores, dtype = float)
    seq_len   = scores.shape[-1]

    future    = np.triu(np.ones((seq_len, seq_len), dtype = bool), k = 1)

    masked    = scores.copy()
    masked[..., future] = mask_value

    return masked