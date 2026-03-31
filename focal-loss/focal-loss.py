import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)
    
    # Compute focal loss
    loss = - (y * ((1 - p) ** gamma) * np.log(p) +
              (1 - y) * (p ** gamma) * np.log(1 - p))
    
    # Return mean loss
    return np.mean(loss)