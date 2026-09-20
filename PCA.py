import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        mean = np.mean(X, axis=0) # Compute mean of input data along each feature dimension
        X = X - mean # Subtract mean from input data to center it around 0
        cov = np.cov(X.T) # Compute the covariance matrix of centered input data
        
        # Compute the eigenvectors and eigenvalues of the covarience matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Reverse the order of the eigenvalues and eigenvectors
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:,::-1]

        # Keep only first n_components eigenvectors as principal components
        self.components = eigenvectors[:,:self.n_components]

        # Compute the explained variance ratio for each principal component
        # Compute total variance of input data
        total_variance = np.sum(np.var(X, axis=0))

        # Compute variance explained by each principal component
        self.explained_variances = eigenvalues[:self.n_components]

        # Compute the explained variance ratio for each principal component
        self.explained_variance_ratio_ = self.explained_variances / total_variance

    def transform(self, X):
        # Center the input data around zero using the mean computed during fit step
        X = X - np.mean(X, axis=0)

        # Project the centered input data onto principal components
        transformed_data = np.dot(X, self.components)

        return transformed_data
    
    def fit_transform(self, X):
        self.fit(X)
        transformed_data = self.transform(X)
        return transformed_data