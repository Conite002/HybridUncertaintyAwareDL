import numpy as np
from .base_cp import ConformalPredictor

class EAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, probas=None, labels=None, randomize=False):
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=randomize)

    def compute_scores(self):
        if self.probas is None or self.labels is None or self.u is None:
            raise ValueError("probas, labels, and u must be set.")
        scores = []
        for i in range(len(self.probas)):
            probs = self.probas[i]
            u = self.u[i]
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            prob = 1 - (entropy / np.log(len(probs)))
            sorted_indices = np.argsort(probs)[::-1]
            rank = np.where(sorted_indices == self.labels[i])[0][0] + 1
            score = prob * u if rank == 1 else prob + (rank - self.k_reg + u) * self.lambda_param
            scores.append(score)
        self.scores = np.array(scores)
        return self.scores

    def calibrate(self):
        self.compute_scores()
        self.threshold = np.quantile(self.scores, 1 - self.alpha)
        self.calibrated = True

    def predict(self, probas):
        if not self.calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        pred_sets = []
        for i in range(len(probas)):
            probs = probas[i]
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            prob = 1 - (entropy / np.log(len(probs)))
            u = np.random.uniform() if self.randomize else 0.5
            sorted_indices = np.argsort(probs)[::-1]
            scores = []
            for j, cls in enumerate(sorted_indices):
                rank = j + 1
                score = prob * u if rank == 1 else prob + (rank - self.k_reg + u) * self.lambda_param
                scores.append(score)

            L = 0
            for j in range(len(probs)):
                if scores[j] <= self.threshold:
                    L = j + 1
                else:
                    break
            if L == 0:
                L = 1
            pred_sets.append(list(sorted_indices[:L]))
        return pred_sets
