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
        Calcule le score RAPS pour un échantillon de calibration.
        Selon la formule :
          score = ρ(x,y) + π̂(x)[y]·u + λ·max(r - k_reg, 0)
        où ρ(x,y) est la somme cumulée des probabilités des classes mieux classées que y,
        et r est le rang (1-indexé) de y dans l'ordre décroissant.
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
            score = self._compute_raps_score(self.probas[i], self.labels[i], self.u[i])
            scores.append(score)
        return np.array(scores)
    
    

    def calibrate(self):
        """
        Calibrer le seuil RAPS (tau) à partir des scores.
        """
        scores = self.compute_scores()
        self.threshold = np.quantile(scores, 1 - self.alpha, interpolation='higher')
        self.calibrated = True

    
    def predict(self, probas):
        """
        Génère les ensembles de prédiction conformes pour chaque échantillon,
        en suivant la logique RAPS avec pénalité et randomisation optionnelle.
        """
        if not self.calibrated:
            raise RuntimeError("RAPS is not calibrated. Please call calibrate() before predict().")

        pred_sets = []

        for i in range(probas.shape[0]):
            p = probas[i]
            K = len(p)
            sorted_indices = np.argsort(p)[::-1]
            sorted_probs = p[sorted_indices]
            cum_sum = 0.0
            L = 0

            # 1. Cumulative + penalty loop
            for j in range(K):
                cum_sum += sorted_probs[j]
                penalty = self.lambda_param * max(j - self.k_reg, 0)
                if cum_sum + penalty >= self.threshold:
                    L = j + 1
                    break
            if L == 0:
                L = 1

            # 2. Randomisation (facultatif)
            if self.randomize:
                s_L = sorted_probs[L - 1]
                extra_penalty = self.lambda_param if (L > self.k_reg) else 0
                delta = (cum_sum + extra_penalty - self.threshold) / (s_L + extra_penalty + 1e-12)

                if np.random.uniform(0, 1) >= delta and L > 1:
                    L = L - 1

            pred_set = list(sorted_indices[:L])
            pred_sets.append(pred_set)

        return pred_sets
