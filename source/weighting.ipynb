import numpy as np

def compute_iptw(treatment, propensity_scores):
    """
    Computes Inverse Probability of Treatment Weights (IPTW).
    """
    treatment = np.array(treatment)
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)  # Avoid extreme weights
    weights = np.where(treatment == 1,
                       1 / propensity_scores,
                       1 / (1 - propensity_scores))
    return weights
