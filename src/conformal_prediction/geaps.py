import numpy as np
from .base_cp import ConformalPredictor

class GEAPS(ConformalPredictor):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0,
                 probas=None, labels=None, randomize=False,
                 beta=0.0, gamma=1.0):
        """
        GEAPS (Gap + Entropy Adaptive Prediction Sets) conformal predictor.
        L'entropie remplace le facteur u dans le score.
        """
        super().__init__(alpha=alpha, k_reg=k_reg, lambda_param=lambda_param,
                         probas=probas, labels=labels, randomize=randomize)
        self.beta = beta      # pour extensions futures
        self.gamma = gamma    # pour extensions futures

    def compute_scores(self):
        if self.probas is None or self.labels is None:
            raise ValueError("probas and labels must be set before computing scores.")
        
        scores = []
        for i in range(len(self.probas)):
            probs = self.probas[i]
            sorted_indices = np.argsort(probs)[::-1]
            p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
            gap = p1 - p2
            rank = np.where(sorted_indices == self.labels[i])[0][0] + 1
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            u = 1 - (entropy / np.log(len(probs)))  # normalised entropy → confiance

            if rank == 1:
                score = u * gap
            else:
                score = gap + (rank - self.k_reg + u) * self.lambda_param

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
            probs = probas[i]
            sorted_indices = np.argsort(probs)[::-1]
            p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
            gap = p1 - p2
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            u = 1 - (entropy / np.log(len(probs)))

            scores = []
            for j, cls in enumerate(sorted_indices):
                rank = j + 1
                if rank == 1:
                    score = u * gap
                else:
                    score = gap + (rank - self.k_reg + u) * self.lambda_param
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
