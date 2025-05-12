import numpy as np
from .base_cp import ConformalPredictor


class SAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, randomize=False, labels=None, probas=None):
        """
        SAPS (Sums of Adaptive Prediction Sets) conformal predictor.
        """
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param, randomize=randomize,
                         labels=labels, probas=probas)
        
    def _compute_saps_score(self, probs, true_label, u):
        max_prob = np.max(probs)
        sorted_indices = np.argsort(probs)[::-1]
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        return u*max_prob if rank == 1 else max_prob + (rank - self.k_reg + u) * self.lambda_param
    
    def compute_scores(self):
        """
        Compute the scores based on the predicted probabilities and the true labels.
        """
        scores = []
        n_samples = self.probas.shape[0]
        for i in range(n_samples):
            score = self._compute_saps_score(self.probas[i], self.labels[i], self.u[i])
            scores.append(score)
        self.scores = np.array(scores)
        return self.scores
    
    def calibrate(self):
        self.scores = self.compute_scores()
        self.threshold = np.quantile(self.scores, 1 - self.alpha, interpolation="higher")
        self.calibrated = True
        
    def predict(self, probas): 
        if not self.calibrated:
            raise ValueError("The model must be calibrated before making predictions.")
        pred_sets = []
        K = len(probas[0])
        for i in range(probas.shape[0]):
            sorted_indices = np.argsort(probas[i])[::-1]
            sorted_probs = probas[i][sorted_indices]
            msp = sorted_probs[0]
            
            pred_set = []
            for rank, (cls, p) in enumerate(zip(sorted_indices, sorted_probs), start=1):
                u_list = np.random.uniform(0, 1, size=K) if self.randomize else np.full(K, 0.5)

                if rank == 1:
                    s = msp * u_list[rank - 1]
                else:
                    s = msp + (rank - self.k_reg + u_list[rank - 1]) * self.lambda_param
                    
                pred_set.append(cls)
                if s >= self.threshold:
                    break
            pred_sets.append(pred_set)
        return pred_sets
                