import numpy as np
from .base_cp import ConformalPredictor

class GAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, probas=None, labels=None, randomize=True):
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=randomize)

    def _compute_gaps_score(self, probs, true_label, u):
        sorted_indices = np.argsort(probs)[::-1]
        p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
        gap = p1 - p2
        rank = np.where(sorted_indices == true_label)[0][0] + 1

        if rank == 1:
            score = u * gap
        else:
            score = gap + (rank - self.k_reg + u) * self.lambda_param
        return score

    def compute_scores(self):
        if self.probas is None or self.labels is None:
            raise ValueError("probas and labels must be set before computing scores.")

        scores = []
        for i in range(len(self.probas)):
            u = np.random.uniform(0, 1) if self.randomize else 1.0
            score = self._compute_gaps_score(self.probas[i], self.labels[i], u)
            scores.append(score)

        self.scores = np.array(scores)
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
            p1, p2 = probas[i][sorted_indices[0]], probas[i][sorted_indices[1]]
            gap = p1 - p2
            u = np.random.uniform(0, 1) if self.randomize else 1.0

            pred_set = []
            for rank, cls in enumerate(sorted_indices, start=1):
                if rank == 1:
                    score = u * gap
                else:
                    score = gap + (rank - self.k_reg + u) * self.lambda_param

                pred_set.append(cls)
                if score >= self.threshold:
                    break

            pred_sets.append(pred_set)
        return pred_sets
