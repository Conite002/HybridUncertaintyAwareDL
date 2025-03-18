import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import norm
import torch

class ConformalPredictor:
    def __init__(self, alpha: float = 0.1, method: str = 'aps'):
        """
        Initialize Conformal Predictor
        
        Args:
            alpha: Desired miscoverage level (e.g., 0.1 for 90% coverage)
            method: Conformal prediction method ('aps', 'raps', or 'saps')
        """
        self.alpha = alpha
        self.method = method.lower()
        self.calibration_scores = None
        self.threshold = None
        
    def _compute_scores(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute conformity scores based on the chosen method"""
        n_samples = probabilities.shape[0]
        scores = []
        
        for i in range(n_samples):
            true_class_prob = probabilities[i, labels[i]]
            if self.method == 'aps':
                score = 1 - true_class_prob
            elif self.method == 'raps':
                sorted_probs = -np.sort(-probabilities[i])  # Sort in descending order
                score = sorted_probs[0] - true_class_prob
            elif self.method == 'saps':
                sorted_probs = -np.sort(-probabilities[i])  # Sort in descending order
                margin = sorted_probs[0] - true_class_prob
                score = margin / (sorted_probs[0] - sorted_probs[1] + 1e-10)  # Normalized score
            scores.append(score)
            
        return np.array(scores)
    
    def calibrate(self, cal_probabilities: np.ndarray, cal_labels: np.ndarray):
        """Calibrate the conformal predictor using validation data"""
        self.calibration_scores = self._compute_scores(cal_probabilities, cal_labels)
        n = len(self.calibration_scores)
        self.threshold = np.quantile(self.calibration_scores, 
                                   np.ceil((n + 1) * (1 - self.alpha)) / n,
                                   interpolation='higher')
        
    def predict(self, probabilities: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        Generate prediction sets for test samples
        
        Returns:
            prediction_sets: List of lists containing class indices in each prediction set
            sizes: Array of prediction set sizes
        """
        if self.threshold is None:
            raise ValueError("Calibrate the predictor first using calibrate()")
            
        n_samples, n_classes = probabilities.shape
        prediction_sets = []
        
        for i in range(n_samples):
            pred_set = []
            for j in range(n_classes):
                if self.method == 'aps':
                    score = 1 - probabilities[i, j]
                elif self.method == 'raps':
                    score = np.max(probabilities[i]) - probabilities[i, j]
                elif self.method == 'saps':
                    margin = np.max(probabilities[i]) - probabilities[i, j]
                    score = margin / (np.max(probabilities[i]) - np.sort(probabilities[i])[-2] + 1e-10)
                
                if score <= self.threshold:
                    pred_set.append(j)
            prediction_sets.append(pred_set)
            
        sizes = np.array([len(s) for s in prediction_sets])
        return prediction_sets, sizes
    
    def evaluate(self, prediction_sets: List[List[int]], true_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate the conformal predictor's performance"""
        coverage = np.mean([label in pred_set for label, pred_set in zip(true_labels, prediction_sets)])
        avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])
        
        return {
            'coverage': coverage,
            'average_set_size': avg_size,
            'efficiency': 1.0 / avg_size if avg_size > 0 else 0.0
        } 