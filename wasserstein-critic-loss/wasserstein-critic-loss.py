import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_scores = np.array(real_scores, dtype=float)
    fake_scores = np.array(fake_scores, dtype=float)
    
    # Compute means
    mean_real = np.mean(real_scores)
    mean_fake = np.mean(fake_scores)
    
    # Wasserstein loss
    return float(mean_fake - mean_real)