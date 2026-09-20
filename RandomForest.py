
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

class RandomForest:
     def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 max_leaf_nodes=None,
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.trees = []

     def fit(self, X, y):
        self.trees = []
        rng = np.random.default_rng(self.random_state)
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state
            )

            dataset_sample = self.bootstrap_samples(dataset, rng)
            X_sample, y_sample = dataset_sample[:, :-1], dataset_sample[:, -1]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

     def bootstrap_samples(self, dataset, rng):
        n_samples = dataset.shape[0]
        indices = rng.choice(n_samples, n_samples, replace=True)
        return dataset[indices]

     def most_common_label(self, y):
        y = list(y)
        return max(y, key=y.count)

     def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Handle both single-sample and multi-sample cases
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        preds = predictions.T  # transpose instead of swapaxes
        majority_predictions = np.array([self.most_common_label(pred) for pred in preds])
        return majority_predictions

     def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)