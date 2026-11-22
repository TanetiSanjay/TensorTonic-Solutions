import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:

    scores_copy     = np.array(scores, copy=True)   
    seq_len     = scores_copy.shape[-1]
    mask        = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores_copy = np.where(mask, mask_value, scores_copy)
    

    return scores_copy