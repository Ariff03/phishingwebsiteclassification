import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state, check_array
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR, _libsvm

class SVM:
    def __init__(self, regression=False, C=10.0, kernel='rbf', degree=3, solver='auto', gamma=1.0, epsilon=0.1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, max_iter=-1, random_state=None):
        self.regression = regression
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.solver = solver
        self.gamma = gamma
        self.epsilon = epsilon
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        X = X.astype(np.float64)
        y = y.astype(np.float64).ravel()

        rnd = check_random_state(self.random_state)
        seed = rnd.randint(np.iinfo('i').max)

        if self.gamma == 'scale':
            self.gamma = 1.0 / (X.shape[1] * X.var()) if X.var() != 0 else 1.0
        elif self.gamma == 'auto':
                self.gamma = 1.0 / X.shape[1]
        else:
             self.gamma = self.gamma
        
        if self.solver == 'auto':
             self.solver = 'epsilon_svr' if self.regression else 'c_svc'

        libsvm_impl = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']
        self.solver = libsvm_impl.index(self.solver)
        (self.support_, self.support_vectors_, self._n_support, self.dual_coef_, self.intercept_,
         self._probA, self._probB, self.fit_status_, self._num_iter) = _libsvm.fit(X, y, C=self.C, svm_type=self.solver, kernel=self.kernel, gamma=self.gamma, 
                                                                                   degree=self.degree, epsilon=self.epsilon, coef0=self.coef0, 
                                                                                   tol=self.tol, shrinking=self.shrinking, probability=self.probability, 
                                                                                   cache_size=self.cache_size, max_iter=self.max_iter, random_seed=seed)
    
    def predict(self, X_test):
         X_test = X_test.astype(np.float64)
         prediction = _libsvm.predict(X_test, self.support_, self.support_vectors_, self._n_support,
                                     self.dual_coef_, self.intercept_, self._probA, self._probB,
                                     svm_type=self.solver, kernel=self.kernel, degree=self.degree,
                                     coef0=self.coef0, gamma=self.gamma, cache_size=self.cache_size
                                     )
         return prediction if self.regression else prediction.astype(int)