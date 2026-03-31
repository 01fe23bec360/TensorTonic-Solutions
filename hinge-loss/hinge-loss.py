import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores
    """
    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    
    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("Shapes of y_true and y_score must match")
    
    # Validate labels
    if not np.all((y_true == 1) | (y_true == -1)):
        raise ValueError("y_true must contain only -1 or +1")
    
    # Compute hinge loss
    loss = np.maximum(0, margin - y_true * y_score)
    
    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")