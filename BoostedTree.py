import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BoostedTree:
    def __init__(self, X, gradients, hessians, params, max_depth, idxs=None, feature_idxs=None):
        self.feature_idxs = feature_idxs if feature_idxs is not None else np.arange(X.shape[1])
        self.X = X.values if isinstance(X, pd.DataFrame) else X 
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians
        self.params = params
        self.min_child_weight = self.params['min_child_weight'] if self.params['min_child_weight'] else 1.0
        self._lambda = self.params['reg_lambda'] if self.params['reg_lambda'] else 1.0
        self.gamma = self.params['gamma'] if self.params['gamma'] else 1.0
        self.alpha = self.params['reg_alpha'] if self.params['reg_alpha'] else 0.0
        self.max_depth = max_depth
        self.ridxs = idxs if idxs is not None else np.arange(len(gradients))
        self.num_examples = len(self.ridxs)
        self.num_features = X.shape[1]
        G = self.gradients[self.ridxs].sum()
        H = self.hessians[self.ridxs].sum()
        if abs(G) > self.alpha:
            self.weight = -np.sign(G) * max(abs(G) - self.alpha, 0) / (H + self._lambda)
        else:
            self.weight = 0.0 

        self.split_score = 0.0
        self.split_idx = 0
        self.threshold = 0.0
        self._build_tree_structure()

    def _build_tree_structure(self):
        if self.max_depth <= 0:
            return
        
        for fidx in range(self.num_features):
            self._find_best_split_score(fidx)

        if self._is_leaf:
            return
        
        feature = self.X[self.ridxs, self.split_idx]
        left_idxs = np.nonzero(feature <= self.threshold)[0]
        right_idxs = np.nonzero(feature > self.threshold)[0]

        self.left = BoostedTree(self.X, self.gradients, self.hessians, self.params, self.max_depth - 1, self.ridxs[left_idxs])
        self.right = BoostedTree(self.X, self.gradients, self.hessians, self.params, self.max_depth - 1, self.ridxs[right_idxs])

    def _find_best_split_score(self, fidx):

            feature = self.X[self.ridxs, fidx]
            gradients = self.gradients[self.ridxs]
            hessians = self.hessians[self.ridxs]

            sorted_idxs = np.argsort(feature)
            sorted_feature = feature[sorted_idxs]
            sorted_gradient = gradients[sorted_idxs]
            sorted_hessians = hessians[sorted_idxs]

            hessian_sum = sorted_hessians.sum()
            gradient_sum = sorted_gradient.sum(0)

            right_hessian_sum = hessian_sum
            right_gradient_sum = gradient_sum
            left_hessian_sum = 0.0
            left_gradient_sum = 0.0

            for idx in range(0, self.num_examples - 1):
                candidate = sorted_feature[idx]
                neighbor = sorted_feature[idx + 1]

                gradient = sorted_gradient[idx]
                hessian = sorted_hessians[idx]

                right_gradient_sum -= gradient
                right_hessian_sum -= hessian
                left_gradient_sum += gradient
                left_hessian_sum += hessian

                if right_hessian_sum <= self.min_child_weight:
                    return
                
                right_score = (right_gradient_sum ** 2) / (right_hessian_sum + self._lambda)
                left_score = (left_gradient_sum ** 2) / (left_hessian_sum + self._lambda)
                score_before_split = (gradient_sum ** 2) / (hessian_sum + self._lambda)
                gain = 0.5 * (left_score + right_score - score_before_split) - self.gamma - self.alpha

                if gain > self.split_score:
                    self.split_score = gain
                    self.split_idx = fidx
                    self.threshold = (candidate + neighbor) / 2 
                
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        preds = np.array([self._predict_row(example) for example in X])
        return preds.reshape(-1)  # Predict each row

    def _predict_row(self, example):

        if self._is_leaf:
            return self.weight  # Return leaf weight
        child = self.left if example[self.split_idx] <= self.threshold else self.right
        return child._predict_row(example)  # Recurse down the tree

    @property
    def _is_leaf(self):

        return self.split_score == 0.0  # Leaf node if no gain found

