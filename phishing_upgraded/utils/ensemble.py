"""
Ensemble Voting Module
Combines predictions from multiple models using weighted averaging.
"""

import numpy as np
from typing import Dict, Optional, List

# Default weights (can be tuned to reflect validation AUC)
DEFAULT_WEIGHTS = {
    "DNN":          1.0,
    "CNN1D":        1.0,
    "BiLSTM":       1.0,
    "AttentionDNN": 1.0,
}


def weighted_ensemble(
    predictions: Dict[str, Optional[float]],
    weights: Dict[str, float] = None,
) -> float:
    """
    Compute a weighted average ensemble score.

    Parameters
    ----------
    predictions : {model_name: probability}  –  None entries are skipped
    weights     : {model_name: weight}

    Returns
    -------
    float  –  ensemble phishing probability in [0, 1]
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0
    for name, prob in predictions.items():
        if prob is None:
            continue
        w = weights.get(name, 1.0)
        weighted_sum += prob * w
        total_weight += w

    if total_weight == 0:
        return 0.5
    return float(weighted_sum / total_weight)


def majority_vote(predictions: Dict[str, Optional[float]], threshold: float = 0.5) -> str:
    """
    Hard majority vote across all models.

    Returns 'Phishing' or 'Safe'.
    """
    votes: List[int] = []
    for prob in predictions.values():
        if prob is not None:
            votes.append(1 if prob >= threshold else 0)

    if not votes:
        return "Unknown"
    return "Phishing" if sum(votes) > len(votes) / 2 else "Safe"


def confidence_level(ensemble_prob: float) -> str:
    """Human-readable confidence label."""
    if ensemble_prob >= 0.90 or ensemble_prob <= 0.10:
        return "Very High"
    elif ensemble_prob >= 0.75 or ensemble_prob <= 0.25:
        return "High"
    elif ensemble_prob >= 0.60 or ensemble_prob <= 0.40:
        return "Medium"
    return "Low"
