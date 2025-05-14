import numpy as np
from .base_cp import ConformalPredictor

class GAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, randomize=True, probas=None, labels=None):
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=randomize)

    def compute_scores(self):
        if self.probas is None or self.labels is None or self.u is None:
            raise ValueError("probas, labels, and u must be set before computing scores.")
        
        scores = []
        for i in range(len(self.probas)):
            probs = self.probas[i]
            u = self.u[i]
            sorted_indices = np.argsort(probs)[::-1]
            p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
            gap = p1 - p2
            rank = np.where(sorted_indices == self.labels[i])[0][0] + 1
            score = u * gap if rank == 1 else gap + (rank - self.k_reg + u) * self.lambda_param
            scores.append(score)
        self.scores = np.array(scores)
        return self.scores

    def calibrate(self):
        self.compute_scores()
        # self.threshold = np.quantile(self.scores, 1 - self.alpha)
        print(f"[DEBUG] Scores shape: {np.shape(self.scores)} | Type: {type(self.scores)}")
        sorted_scores = np.sort(self.scores)
        n = len(sorted_scores)
        index = int(np.ceil(n * (1 - self.alpha))) - 1
        index = min(index, n - 1)
        self.threshold = sorted_scores[index]
        self.calibrated = True

    def predict(self, probas):
        if not self.calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        pred_sets = []
        for i in range(len(probas)):
            p = probas[i]
            K = len(p)
            sorted_indices = np.argsort(p)[::-1]
            p1, p2 = p[sorted_indices[0]], p[sorted_indices[1]]
            gap = p1 - p2
            u_list = np.random.uniform(0, 1, K) if self.randomize else np.full(K, 0.5)

            scores = []
            for j in range(K):
                rank = j + 1
                score = u_list[j] * gap if rank == 1 else gap + (rank - self.k_reg + u_list[j]) * self.lambda_param
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
