import numpy as np

class LinearBooster:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def train(self, X, gradients, hessians, lr):
        # Linear XGBoost uses Newton update:
        # w = w - grad / hess
        g = gradients.sum(axis=0) if gradients.ndim > 1 else np.dot(gradients, X)
        h = hessians.sum()

        if h == 0:
            return

        # weight update (Newton step)
        self.weights -= lr * g / h

        # bias update
        self.bias -= lr * gradients.sum() / h

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias