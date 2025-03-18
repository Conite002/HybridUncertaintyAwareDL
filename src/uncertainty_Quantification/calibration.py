import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).

    Args:
        y_true (np.array): True labels.
        y_prob (np.array): Model confidence scores.
        n_bins (int): Number of bins.

    Returns:
        ece (float): Expected Calibration Error.
    """
    if y_prob.ndim != 2:
        raise ValueError(f"`y_prob` should be a 2D array of shape (num_samples, num_classes), but got shape {y_prob.shape}")

    if y_true.ndim != 1:
        raise ValueError(f"`y_true` should be a 1D array of shape (num_samples,), but got shape {y_true.shape}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob.max(axis=1), bin_edges) - 1  # Use max confidence

    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            acc = np.mean(y_true[bin_mask] == np.argmax(y_prob[bin_mask], axis=1))
            conf = np.mean(y_prob[bin_mask])
            ece += np.abs(acc - conf) * np.sum(bin_mask) / len(y_true)
    
    return ece



def reliability_diagram(y_true, y_prob, n_bins=10):
    """
    Plots a reliability diagram.

    Args:
        y_true (np.array): True labels (1D array).
        y_prob (np.array): Model probability scores (2D array with shape [num_samples, num_classes]).
        n_bins (int): Number of bins.

    Returns:
        None (Displays a reliability diagram).
    """
    # ✅ Ensure `y_prob` is a 2D probability matrix
    if y_prob.ndim != 2:
        raise ValueError(f"`y_prob` should be a 2D array of shape (num_samples, num_classes), but got shape {y_prob.shape}")

    # ✅ Extract max probability per sample (model confidence)
    max_confidence = y_prob.max(axis=1)  # Get highest probability per sample

    # ✅ Compute Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true == np.argmax(y_prob, axis=1), max_confidence, n_bins=n_bins)

    # ✅ Plot Reliability Diagram
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid()
    plt.show()
