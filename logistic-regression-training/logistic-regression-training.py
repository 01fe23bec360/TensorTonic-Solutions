import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    # Training loop
    for _ in range(steps):
        # Linear combination
        z = X @ w + b
        
        # Prediction
        p = _sigmoid(z)
        
        # Gradients
        dw = (1 / N) * (X.T @ (p - y))
        db = (1 / N) * np.sum(p - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b