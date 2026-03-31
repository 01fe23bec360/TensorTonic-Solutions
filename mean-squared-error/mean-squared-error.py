import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)
    
    # Check shape
    if y_pred.shape != y_true.shape:
        return None
    
    # Compute MSE
    mse = np.mean((y_pred - y_true) ** 2)
    
    return float(mse)
