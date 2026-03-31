import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    p = np.array(predictions, dtype=float)
    y = np.array(targets, dtype=float)
    
    # Compute p_t
    p_t = y * p + (1 - y) * (1 - p)
    
    # Compute focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)
    
    # Return mean loss
    return np.mean(loss)