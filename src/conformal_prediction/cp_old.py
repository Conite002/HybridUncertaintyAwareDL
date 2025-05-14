import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ConformalPredictor:
    def __init__(self, method='raps', lambda_param=1.0, k_reg=1, randomize=True, dynamic_threshold_fn=None):
        """
        Paramètres :
          - method : 'raps' ou 'saps'
          - lambda_param : hyperparamètre de régularisation (λ)
          - k_reg : paramètre de seuil pour la régularisation (k₍reg₎)
          - randomize : si True, on utilise la randomisation (u ~ U(0,1))
        """
        self.method = method.lower()
        self.lambda_param = lambda_param
        self.k_reg = k_reg
        self.randomize = randomize
        self.tau = None
        self.dynamic_threshold_fn = dynamic_threshold_fn

        

    def _compute_aps_score(self, probs, true_label, u=None):

        sorted_indices = np.argsort(probs)[::-1]
        pos = np.where(sorted_indices == true_label)[0][0]
        score = np.sum(probs[sorted_indices[:pos]]) + u*probs[true_label] 
        
        return score

    
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
        score = rho + probs[true_label] * u + self.lambda_param * max(rank - self.k_reg, 0)
        
        return score

    def _compute_saps_score(self, probs, true_label, u):
        """
        Calcule le score SAPS pour un échantillon de calibration.
        Pour la classe ayant le rang 1, score = u * MSP.
        Sinon, score = MSP + (r - 2 + u)*λ,
        où MSP est la valeur maximale de la softmax, et r est le rang de la vraie classe.
        """
        max_prob = np.max(probs)
        sorted_indices = np.argsort(probs)[::-1]
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        if rank == 1:
            score = u * max_prob
        else:
            score = max_prob + (rank - self.k_reg + u) * self.lambda_param
        return score

    def _compute_gaps_score(self, probs, true_label, u):
        """
        Compute GAP-based calibration score for conformal calibration.

        Logic:
        - For top-ranked class (rank=1), score = u * ConfidenceGap
        - For classes with rank > 1:
            score = ConfidenceGap + (rank - k_reg + u) * lambda_param

        Args:
            probs (array): predicted class probabilities.
            true_label (int): true class label.
            u (float): randomization uniform(0,1).

        Returns:
            float: GAP-based calibration score.
        """
        sorted_indices = np.argsort(probs)[::-1]
        p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
        gap = p1 - p2
        rank = np.where(sorted_indices == true_label)[0][0] + 1

        if rank == 1:
            score = u * gap
        else:
            score = gap + (rank - self.k_reg + u) * self.lambda_param

        return score

    def compute_geaps_score(self, probs, true_label, u):
        """
        Compute GEAPS score (Gaps + Entropy) for a calibration sample.

        Replaces random u by normalized entropy.

        Args:
            probs (np.ndarray): Probability vector (shape: [K])
            true_label (int): Ground-truth class

        Returns:
            float: Conformal score
        """
        K = len(probs)
        sorted_indices = np.argsort(probs)[::-1]
        p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
        gap = p1 - p2
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        u = 1-(entropy / np.log(K))
        if rank == 1:
            score = u * gap
        else:
            score = gap + (rank - self.k_reg + u) * self.lambda_param
        return score
    
    def _compute_eaps_score(self, probs, true_label, u):
        
        K = len(probs)
        sorted_indices = np.argsort(probs)[::-1]
        p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
        gap = p1 - p2
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        prob = 1-(entropy / np.log(K))
        if rank == 1:
            score = prob
        else:
            score = prob + (rank - self.k_reg + u) * self.lambda_param
        return score

        
    def calibrate(self, cal_probs, cal_labels, cal_yvars, alpha=0.1):
        scores = []
        n = len(cal_labels)
        for i in range(n):
            u = np.random.uniform(0, 1) if self.randomize else 0.5
            if self.method == 'raps':
                score = self._compute_raps_score(cal_probs[i], cal_labels[i], u)
            elif self.method == 'saps':
                score = self._compute_saps_score(cal_probs[i], cal_labels[i], u)
            elif self.method == 'aps':
                score = self._compute_aps_score(cal_probs[i], cal_labels[i], u)
            elif self.method == 'gaps':
                score = self._compute_gaps_score(cal_probs[i], cal_labels[i], u)
            elif self.method == 'geaps':
                score = self.compute_geaps_score(cal_probs[i], cal_labels[i], u)
            elif self.method == 'eaps':
                score = self._compute_eaps_score(cal_probs[i], cal_labels[i], u)
            else:
                raise ValueError("Méthode non supportée : " + self.method)
            scores.append(score)
        scores = np.array(scores)
        sorted_scores = np.sort(scores)
        index = int(np.ceil((n + 1) * (1 - alpha))) - 1
        index = min(index, n - 1)
        self.tau = sorted_scores[index]
        if self.method == 'raps':
            print(f"[DEBUG] tau: {self.tau} | index: {index} | n: {n} | alpha: {alpha} | sorted_scores: {sorted_scores}")
        return self.tau

    def _extract_uncertainty_feature(self, probs):
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return entropy / np.log(len(probs))


    def predict(self, probs):
        # === 1. Threshold dynamique ou statique ===
        if self.dynamic_threshold_fn is not None:
            input_feat = self._extract_uncertainty_feature(probs)
            threshold = self.dynamic_threshold_fn(torch.tensor([[input_feat]], dtype=torch.float32))
        else:    
            if self.tau is None:
                raise ValueError("Veuillez d'abord calibrer le prédicteur (méthode calibrate).")
            threshold = self.tau

        K = len(probs)
        
        if self.method == 'raps':
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cum_sum = 0.0
            L = 0
            for j in range(K):
                cum_sum += sorted_probs[j]
                penalty = self.lambda_param * max(j - self.k_reg, 0)
                if cum_sum + penalty > threshold:    
                    L = j + 1
                    break
            if L == 0:
                L = 1
            if self.randomize:
                s_L = sorted_probs[L - 1]
                extra_penalty = self.lambda_param if (L > self.k_reg) else 0
                delta = (cum_sum + extra_penalty - threshold) / (s_L + extra_penalty + 1e-12)

                if np.random.uniform(0, 1) >= delta and L > 1:
                    L = L - 1
            pred_set = sorted_indices[:L]

        elif self.method == 'saps':
            sorted_indices = np.argsort(probs)[::-1]
            max_prob = np.max(probs)
            u_list = np.random.uniform(0, 1, size=K) if self.randomize else np.full(K, 0.5)
            scores = []
            for j, idx in enumerate(sorted_indices):
                rank = j + 1
                if rank == 1:
                    score = u_list[j] * max_prob
                else:
                    score = max_prob + (rank - self.k_reg + u_list[j]) * self.lambda_param
                scores.append(score)
            scores = np.array(scores)
            L = 0
            for j in range(K):
                if scores[j] <= threshold:
                    L = j + 1
                else:
                    break
                
            if L == 0:
                L = 1
            pred_set = sorted_indices[:L]
            
        elif self.method == 'aps':
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cum_sum = 0.0
            L = 0
            for j in range(K):
                cum_sum += sorted_probs[j]
                if cum_sum >= threshold:
                    L = j + 1
                    break
            

            pred_set = sorted_indices[:L]
            
            
        elif self.method == 'gaps':
            sorted_indices = np.argsort(probs)[::-1]
            p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
            gap = p1 - p2

            u_list = np.random.uniform(0, 1, size=K) if self.randomize else np.full(K, 0.5)
            scores = []

            for j, idx in enumerate(sorted_indices):
                rank = j + 1
                if rank == 1:
                    score = u_list[j] * gap
                else:
                    score = gap + (rank - self.k_reg + u_list[j]) * self.lambda_param
                scores.append(score)

            scores = np.array(scores)

            L = 0
            for j in range(K):
                if scores[j] <= self.tau:
                    L = j + 1
                else:
                    break

            if L == 0:
                L = 1  
            pred_set = sorted_indices[:L]
        
        elif self.method == 'geaps':
            sorted_indices = np.argsort(probs)[::-1]
            p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
            gap = p1 - p2
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            prob = 1 - (entropy / np.log(K))
            scores = []
            for j, idx in enumerate(sorted_indices):
                rank = j + 1
                if rank == 1:
                    score = prob * gap
                else:
                    score = gap + (rank - self.k_reg + prob) * self.lambda_param
                scores.append(score)
            scores = np.array(scores)
            L = 0
            for j in range(K):
                if scores[j] <= self.tau:
                    L = j + 1
                else:
                    break
            if L == 0:
                L = 1
            pred_set = sorted_indices[:L]

        elif self.method == 'eaps':
            u_list = np.random.uniform(0, 1, size=K) if self.randomize else np.full(K, 0.5)
            
            sorted_indices = np.argsort(probs)[::-1]
            p1, p2 = probs[sorted_indices[0]], probs[sorted_indices[1]]
            gap = p1 - p2
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            prob = 1 - (entropy / np.log(K))
            scores = []
            for j, idx in enumerate(sorted_indices):
                rank = j + 1
                if rank == 1:
                    score = prob
                else:
                    score = prob + (rank - self.k_reg + u_list[j]) * self.lambda_param
                scores.append(score)
            scores = np.array(scores)
            L = 0
            for j in range(K):
                if scores[j] <= self.tau:
                    L = j + 1
                else:
                    break
            if L == 0:
                L = 1
            pred_set = sorted_indices[:L]

        else:
            raise ValueError("Méthode non supportée : " + self.method)
            
        return pred_set, sorted_indices
    