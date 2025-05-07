import numpy as np
from .base_cp import ConformalPredictor



class RAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, randomize=False, probas=None, labels=None):
        """
        RAPS (Regularized Adaptive Prediction Sets) conformal predictor.
        """
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         randomize=randomize, probas=probas, labels=labels)

        
        
    def _compute_raps_score(self, probs, true_label, u):
        """
        Score = somme des classes mieux classées + pi(y) * u + régularisation
        """
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        rho = 0.0 if rank == 1 else np.sum(sorted_probs[:rank-1])
        penalty = self.lambda_param * max(rank - self.k_reg, 0)
        score = rho + probs[true_label] * u + penalty
        return score
    
    def compute_scores(self):
        """
        Compute the scores based on the predicted probabilities and the true labels.
        """
        scores = []
        n_samples = self.probas.shape[0]
        for i in range(n_samples):
            u = np.random.uniform(0, 1) if self.randomize else 1.0
            score = self._compute_raps_score(self.probas[i], self.labels[i], u)
            scores.append(score)
        return np.array(scores)
    
    

    def calibrate(self):
        """
        Calibrer le seuil RAPS (tau) à partir des scores.
        """
        scores = self.compute_scores(self.probas, self.labels)
        self.threshold = np.quantile(scores, 1 - self.alpha, interpolation='higher')
        self.calibrated = True
        
    def predict(self, probas):
        """
        Génère les ensembles de prédiction pour chaque échantillon.
        """
        
        if not self.calibrated:
            raise RuntimeError("RAPS is not calibrated. Please call calibrate() before predict().")
        
        pred_sets = []
        sorted_indices = np.argsort(probas, axis=1)[:, ::-1]
        sorted_probas = np.take_along_axis(probas, sorted_indices, axis=1)
        
        for i in range(probas.shape[0]):
            cum_sum = 0.0
            pred_set = []
            for j, p in enumerate(sorted_probas[i]):
                cum_sum += p
                pred_set.append(sorted_indices[i][j])
                if cum_sum >= self.threshold:
                    break
            pred_sets.append(pred_set)
        return pred_sets
    