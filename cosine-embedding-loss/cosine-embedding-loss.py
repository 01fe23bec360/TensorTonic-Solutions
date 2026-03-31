import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    
    # Compute cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    
    cos_sim = dot / (norm1 * norm2)
    
    # Compute loss based on label
    if label == 1:
        return float(1 - cos_sim)
    elif label == -1:
        return float(max(0, cos_sim - margin))
    else:
        raise ValueError("label must be +1 or -1")