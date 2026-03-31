import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # Add epsilon to q for numerical stability
    q = q + eps
    
    # Mask where p > 0 (to avoid log(0) issues)
    mask = p > 0
    
    # Compute KL divergence
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    return kl