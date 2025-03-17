import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score





def rejection_plot(y_true, y_pred, uncertainties, bins=10):
    """
    Generates a rejection plot showing accuracy as we reject uncertain samples.

    Args:
        y_true (np.array): True labels (1D array).
        y_pred (np.array): Model predictions (1D array of class labels).
        uncertainties (np.array): Uncertainty scores for each sample.
        bins (int): Number of bins for rejection.

    Returns:
        None (Displays a rejection plot).
    """
    # ✅ Ensure y_pred is 1D (class labels)
    if y_pred.ndim != 1:
        raise ValueError(f"`y_pred` should be a 1D array of class labels, but got shape {y_pred.shape}")

    # ✅ Ensure all arrays have the same shape
    if y_true.shape != y_pred.shape or y_true.shape != uncertainties.shape:
        raise ValueError("Mismatch in shape between `y_true`, `y_pred`, and `uncertainties`.")

    # Sort by uncertainty (lowest to highest)
    sorted_indices = np.argsort(uncertainties)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    thresholds = np.linspace(0, 1, bins)
    accuracies = []

    for t in thresholds:
        kept_indices = uncertainties <= t  # Keep samples below threshold
        if np.sum(kept_indices) > 0:
            acc = accuracy_score(y_true_sorted[kept_indices], y_pred_sorted[kept_indices])
        else:
            acc = 0  # No samples left
        accuracies.append(acc)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accuracies, marker='o', linestyle='-')
    plt.xlabel("Rejection Threshold (Uncertainty)")
    plt.ylabel("Accuracy")
    plt.title("Rejection Plot: Accuracy vs. Uncertainty")
    plt.grid()
    plt.show()
