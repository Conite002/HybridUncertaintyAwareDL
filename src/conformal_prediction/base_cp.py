# base_cp.py

from abc import ABC, abstractmethod

class ConformalPredictor(ABC):
    def __init__(self, alpha=0.1, k_reg=1, lambda_param=0.0, randomize=False, labels=None, probas=None):
        """
        Base class for conformal predictors.
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.randomize = randomize
        self.k_reg = k_reg
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.probas = probas
        self.labels = labels
        self.threshold = None
        self.calibrated = False
        self.u = None
        self.scores = None

    @abstractmethod
    def compute_scores(self):
        """
        Compute the scores based on the predicted probabilities and the true labels.
        """
        pass

    @abstractmethod
    def calibrate(self):
        pass

    @abstractmethod
    def predict(self, probas):
        pass

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.calibrated = False

    def set_lambda_param(self, lambda_param):
        self.lambda_param = lambda_param
        self.calibrated = False

    def set_k_reg(self, k_reg):
        self.k_reg = k_reg
        self.calibrated = False

    def set_labels(self, labels):
        self.labels = labels
        self.calibrated = False

    def set_probas(self, probas):
        self.probas = probas
        self.calibrated = False
