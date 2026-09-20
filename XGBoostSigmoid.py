import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from XGBoost import XGBoost
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class XGBoostSigmoid:
    def __init__(self, params, threshold=0.5, seed=42):
        self.params = params
        self.threshold = threshold
        self.objective = BinaryCrossEntropyLoss()
        self.base=XGBoost(self.params, self.objective, seed)

    def train(self, X, y, num_rounds):
        self.base.fit(X, y, num_rounds)
    
    def predict(self, X, with_labels=False, threshold=0.5):
        logits = self.base.predict(X)
        probs = self.objective.sigmoid(logits)
        if with_labels:
            return probs, (probs >= self.threshold).astype(int)
        return probs

class BinaryCrossEntropyLoss:
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -709, 709)  # prevent overflow in exp
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def loss(labels, predictions):
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)

        #To avoid log(0)
        epsilon = 1e-15

        probs = np.clip(probs, epsilon, 1 - epsilon)

        #Binary log loss
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    
    @staticmethod
    def gradients(labels, predictions):
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)

        # Gradient of binary cross entropy
        return probs - labels
    
    @staticmethod
    def hessians(labels, predictions):
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)

        # Gradient of binary cross entropy
        return probs * (1 - probs)