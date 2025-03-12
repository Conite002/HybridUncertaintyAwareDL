import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from scipy import stats
import torch
from typing import Dict, Tuple, List, Union

class UncertaintyMetrics:
    @staticmethod
    def expected_calibration_error(
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error
        
        Args:
            probabilities: Predicted probabilities (N, C)
            labels: True labels (N,)
            num_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        pred_labels = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        # Create bins and compute ECE
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(pred_labels[in_bin] == labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * np.sum(in_bin) / len(labels)
                
        return float(ece)
    
    @staticmethod
    def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Brier Score
        
        Args:
            probabilities: Predicted probabilities (N, C)
            labels: True labels (N,)
            
        Returns:
            Brier Score
        """
        n_classes = probabilities.shape[1]
        one_hot_labels = np.zeros_like(probabilities)
        one_hot_labels[np.arange(len(labels)), labels] = 1
        return np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
    
    @staticmethod
    def compute_effect_sizes(
        correct_uncertainties: np.ndarray,
        incorrect_uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistical effect sizes
        
        Args:
            correct_uncertainties: Uncertainty values for correct predictions
            incorrect_uncertainties: Uncertainty values for incorrect predictions
            
        Returns:
            Dictionary containing KS statistic and Cohen's d
        """
        # Kolmogorov-Smirnov test
        ks_stat, _ = stats.ks_2samp(correct_uncertainties, incorrect_uncertainties)
        
        # Cohen's d
        n1, n2 = len(correct_uncertainties), len(incorrect_uncertainties)
        var1, var2 = np.var(correct_uncertainties, ddof=1), np.var(incorrect_uncertainties, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (np.mean(incorrect_uncertainties) - np.mean(correct_uncertainties)) / pooled_se
        
        return {
            'ks_statistic': float(ks_stat),
            'cohens_d': float(cohens_d)
        }
    
    @staticmethod
    def compute_ood_metrics(
        id_uncertainties: np.ndarray,
        ood_uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute OOD detection metrics
        
        Args:
            id_uncertainties: Uncertainty values for in-distribution samples
            ood_uncertainties: Uncertainty values for out-of-distribution samples
            
        Returns:
            Dictionary containing AUROC and AUPRC
        """
        # Create labels (0 for ID, 1 for OOD)
        labels = np.concatenate([np.zeros(len(id_uncertainties)),
                               np.ones(len(ood_uncertainties))])
        scores = np.concatenate([id_uncertainties, ood_uncertainties])
        
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        
        return {
            'auroc': float(auroc),
            'auprc': float(auprc)
        }

class ClassificationMetrics:
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            predictions: Predicted labels
            labels: True labels
            probabilities: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing various classification metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions, average='weighted')),
            'recall': float(recall_score(labels, predictions, average='weighted')),
            'f1': float(f1_score(labels, predictions, average='weighted'))
        }
        
        if probabilities is not None:
            try:
                metrics['auc'] = float(roc_auc_score(labels, probabilities, multi_class='ovr'))
            except ValueError:
                metrics['auc'] = float('nan')
                
        return metrics

def compute_spearman_correlation(
    uncertainties: np.ndarray,
    set_sizes: np.ndarray
) -> float:
    """
    Compute Spearman rank correlation between uncertainties and set sizes
    
    Args:
        uncertainties: Array of uncertainty values
        set_sizes: Array of prediction set sizes
        
    Returns:
        Spearman correlation coefficient
    """
    return float(stats.spearmanr(uncertainties, set_sizes)[0]) 