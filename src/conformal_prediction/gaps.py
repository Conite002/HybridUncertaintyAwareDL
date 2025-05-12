import numpy as np
from .base_cp import ConformalPredictor
import numpy as np
from .base_cp import ConformalPredictor


class GAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, randomize=True, probas=None, labels=None):
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=randomize)

    def _compute_gaps_score(self, probs, true_label, u):
        sorted_indices = np.argsort(probs)[::-1]
        p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
        gap = p1 - p2
        rank = np.where(sorted_indices == true_label)[0][0] + 1

        if rank == 1:
            return u * gap
        else:
            return gap + (rank - self.k_reg + u) * self.lambda_param

    def compute_scores(self):
        if self.probas is None or self.labels is None:
            raise ValueError("probas and labels must be set before computing scores.")

        scores = []
        for i in range(len(self.probas)):
            u = np.random.uniform(0, 1) if self.randomize else 0.5
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
            raise RuntimeError("GAPS is not calibrated. Please call calibrate() before predict().")

        pred_sets = []

        for i in range(probas.shape[0]):
            p = probas[i]
            K = len(p)
            sorted_indices = np.argsort(p)[::-1]
            p1, p2 = p[sorted_indices[0]], p[sorted_indices[1]]
            gap = p1 - p2

            # Génération des scores par rang
            u_list = np.random.uniform(0, 1, size=K) if self.randomize else np.full(K, 0.5)
            scores = []

            for j in range(K):
                rank = j + 1
                if rank == 1:
                    score = u_list[j] * gap
                else:
                    score = gap + (rank - self.k_reg + u_list[j]) * self.lambda_param
                scores.append(score)

            scores = np.array(scores)

            # Trouver L : plus grand j tel que score <= tau
            L = 0
            for j in range(K):
                if scores[j] <= self.threshold:
                    L = j + 1
                else:
                    break

            if L == 0:
                L = 1  # au moins un élément

            pred_set = list(sorted_indices[:L])
            pred_sets.append(pred_set)

        return pred_sets
