import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)
    
    # Compute similarity matrix
    S = (Z1 @ Z2.T) / temperature   # shape (N, N)
    
    # Numerical stability: subtract row-wise max
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    
    # Compute exp
    exp_S = np.exp(S_stable)
    
    # Denominator: sum over rows
    denom = np.sum(exp_S, axis=1)
    
    # Numerator: diagonal elements
    num = np.diag(exp_S)
    
    # Compute loss
    loss = -np.log(num / denom)
    
    return np.mean(loss)