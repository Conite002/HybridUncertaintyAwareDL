import numpy as np
from .base_cp import ConformalPredictor

class EAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, probas=None, labels=None):
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=False)

    def _compute_eaps_score(self, probs, true_label):
        K = len(probs)
        sorted_indices = np.argsort(probs)[::-1]
        rank = np.where(sorted_indices == true_label)[0][0] + 1

        entropy = -np.sum(probs * np.log(probs + 1e-12))
        prob = 1 - (entropy / np.log(K))  # Normalized entropy

        if rank == 1:
            return prob
        else:
            return prob + (rank - self.k_reg) * self.lambda_param

    def compute_scores(self):
        if self.probas is None or self.labels is None:
            raise ValueError("probas and labels must be set before computing scores.")

        self.scores = np.array([
            self._compute_eaps_score(self.probas[i], self.labels[i])
            for i in range(len(self.probas))
        ])
        return self.scores

    def calibrate(self):
        self.compute_scores()
        self.threshold = np.quantile(self.scores, 1 - self.alpha, interpolation="higher")
        self.calibrated = True

    def predict(self, probas):
        if not self.calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        pred_sets = []
        for i in range(len(probas)):
            sorted_indices = np.argsort(probas[i])[::-1]
            entropy = -np.sum(probas[i] * np.log(probas[i] + 1e-12))
            prob = 1 - (entropy / np.log(len(probas[i])))

            pred_set = []
            for rank, cls in enumerate(sorted_indices, start=1):
                if rank == 1:
                    score = prob
                else:
                    score = prob + (rank - self.k_reg ) * self.lambda_param

                pred_set.append(cls)
                if score >= self.threshold:
                    break

            pred_sets.append(pred_set)
        return pred_sets
