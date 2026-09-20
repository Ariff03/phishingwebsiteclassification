import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from BoostedTree import BoostedTree
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def default_none():
        return None

class XGBoost:
    def __init__(self, params, objective, seed=42):
        self.trees = []
        self.params = defaultdict(default_none, params)
        self.objective = objective
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.base_score = self.params['base_score'] if self.params['base_score'] else 0.5
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 1e-1
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
        self.colsample_bytree = self.params['colsample_bytree'] if self.params['colsample_bytree'] else 1.0
        self.gamma = self.params['gamma'] if self.params['gamma'] else 0.0
        self.reg_lambda = self.params['lambda'] if self.params['lambda'] else 1.0
        self.reg_alpha = self.params['alpha'] if self.params['alpha'] else 0.0
        self.rng = np.random.default_rng(seed=seed)
        self.dart_rate = self.params['dart_rate'] if self.params['dart_rate'] else 0.1
    
    def fit(self, X, y, num_rounds):
        y = np.array(y).reshape(-1)
        predictions = np.full(len(y), self.base_score, dtype=float)
        for rnd in range(num_rounds):
            gradients = self.objective.gradients(y, predictions)
            hessians = self.objective.hessians(y, predictions)
            # Row sampling
            idxs = None if self.subsample == 1.0 else self.rng.choice(len(y), size=math.floor(self.subsample * len(y)), replace=False)
            # Train one tree on current gradients
            feature_idxs = None
            if self.colsample_bytree < 1.0:
                n_features = X.shape[1]
                n_selected = math.floor(self.colsample_bytree * n_features)
                feature_idxs = self.rng.choice(n_features, size=n_selected, replace=False)
                X_subset = X[:, feature_idxs]
            else:
                X_subset = X
                feature_idxs = np.arange(X.shape[1])    

            tree = BoostedTree(X=X_subset, gradients=gradients, hessians=hessians, params=self.params, max_depth=self.max_depth, idxs=idxs, feature_idxs=feature_idxs)
            self.trees.append(tree)
            
        tree_preds = tree.predict(X).reshape(-1)
        predictions += self.learning_rate * tree_preds
    
    def predict(self, X):
        return self.base_score + np.sum([self.learning_rate * tree.predict(X).reshape(-1) for tree in self.trees], axis=0)