import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Convert to float arrays
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)
    
    # Flatten (works for both 1D and 2D)
    p = p.reshape(-1)
    y = y.reshape(-1)
    
    # Compute intersection
    intersection = np.sum(p * y)
    
    # Compute sums
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    
    # Dice coefficient
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)
    
    # Dice loss
    return 1 - dice