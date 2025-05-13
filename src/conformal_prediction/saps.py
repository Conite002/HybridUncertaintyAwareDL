import numpy as np
from .base_cp import ConformalPredictor

class SAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=2, lambda_param=1.0, randomize=True, probas=None, labels=None):
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=randomize)

    def compute_scores(self):
        if self.probas is None or self.labels is None or self.u is None:
            raise ValueError("probas, labels, and u must be set before computing scores.")

        scores = []
        for i in range(len(self.probas)):
            probs = self.probas[i]
            u = self.u[i]
            max_prob = np.max(probs)
            sorted_indices = np.argsort(probs)[::-1]
            rank = np.where(sorted_indices == self.labels[i])[0][0] + 1
            score = u * max_prob if rank == 1 else max_prob + (rank - 2 + u) * self.lambda_param
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
        K = len(probas[0])

        for i in range(len(probas)):
            probs = probas[i]
            sorted_indices = np.argsort(probs)[::-1]
            max_prob = np.max(probs)
            u_list = np.random.uniform(0, 1, size=K) if self.randomize else np.full(K, 0.5)

            scores = []
            for j in range(K):
                rank = j + 1
                score = u_list[j] * max_prob if rank == 1 else max_prob + (rank - 2 + u_list[j]) * self.lambda_param
                scores.append(score)

            L = 0
            for j in range(K):
                if scores[j] <= self.threshold:
                    L = j + 1
                else:
                    break
            if L == 0:
                L = 1
            pred_sets.append(list(sorted_indices[:L]))
        return pred_sets
