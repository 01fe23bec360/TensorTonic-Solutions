import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Number of samples
    N = y_true.shape[0]
    
    # Select correct class probabilities
    correct_probs = y_pred[np.arange(N), y_true]
    
    # Compute loss
    loss = -np.log(correct_probs)
    
    # Return mean loss
    return np.mean(loss)