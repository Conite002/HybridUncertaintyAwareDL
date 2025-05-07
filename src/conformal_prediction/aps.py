import numpy as np
from .base_cp import ConformalPredictor

class APS(ConformalPredictor):
    def __init__(self, alpha=0.1, randomize=False, probas=None, labels=None):
        super().__init__(alpha=alpha, randomize=randomize, probas=probas, labels=labels)

    def compute_scores(self):
        """
        APS score = somme des probas jusqu’au vrai label (inclus),
        avec randomisation optionnelle sur le dernier terme.
        """
        sorted_indices = np.argsort(self.probas, axis=1)[:, ::-1]
        n_samples = self.probas.shape[0]
        scores = []

        for i in range(n_samples):
            label = self.labels[i]
            sorted_p = self.probas[i][sorted_indices[i]]
            sorted_classes = sorted_indices[i]
            rank = np.where(sorted_classes == label)[0][0]

            if self.randomize:
                u = np.random.uniform(0, 1)
                score = np.sum(sorted_p[:rank]) + u * sorted_p[rank]
            else:
                score = np.sum(sorted_p[:rank+1])

            scores.append(score)

        return np.array(scores)

    def calibrate(self):
        """
        Calibrage : calcule le quantile (1 - alpha) des scores.
        """
        self.scores = self.compute_scores()
        self.threshold = np.quantile(self.scores, 1 - self.alpha, interpolation='higher')
        self.calibrated = True

    def predict(self, probas):
        """
        Génère les ensembles de prédiction en ajoutant les classes jusqu'à dépasser le seuil.
        """
        if not self.calibrated:
            raise ValueError("The model must be calibrated before making predictions.")

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
