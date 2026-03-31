import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute label smoothing cross-entropy loss.
    """
    p = np.array(predictions, dtype=float)
    
    K = len(p)
    
    # Build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)
    
    # Compute cross-entropy loss
    loss = -np.sum(q * np.log(p))
    
    return float(loss)