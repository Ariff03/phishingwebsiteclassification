import numpy as np
import pandas as pd
from RandomForest import RandomForest
from KNN import KNN
from sklearn.tree import DecisionTreeClassifier
from SVM import SVM
from PCA import PCA
from XGBoostSigmoid import XGBoostSigmoid
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import seaborn as sns
from DecisionTree import DecisionTree
from matplotlib.ticker import MaxNLocator

data = pd.read_csv("PhishingData2.csv")
label_mapping = {'Phishing': -1, 'Legitimate': 1}
y,levels = pd.factorize(data['Result']) # Get the class labels (phishing or legit) for confusion matrix from dataset
data['Result'] = data['Result'].map(label_mapping)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# def accuracy(y_true, y_pred):
    # accuracy = np.sum(y_true == y_pred) / len(y_true)
    # return accuracy

# rf = RandomForest()
# rf.fit(X_train.to_numpy(), y_train)
# predictions = rf.predict(X_test)

# acc = accuracy(y_test, predictions)
# print(acc)

def accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return 100 * (correct_predictions / total_samples)

def train_test_split(X, y, random_state=42, test_size=0.3):
     # Get number of samples
    n_samples = X.shape[0]

    # Set the seed for the random number generator
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # Determine the size of the test set
    test_size = int(n_samples * test_size)

    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test    

# Data preprocessing
X = data.iloc[:, :-1].values
features_name = X.tolist()

# Principal Component Analysis (PCA)
# pca = PCA(15)
# pca.fit(X)
# X_transformed = pca.transform(X)
# X_transformed[:,1].shape
# X = X_transformed

y = data.iloc[:, -1].values.reshape(-1,1)

# print(f"Resampled dataset shape: {Counter(y_resampled)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

y_train = np.where(y_train == -1, 0, 1)
y_test = np.where(y_test == -1, 0, 1)

print(y_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

colsample_bytree = list(np.arange(0.1, 1.0, 0.1))
train_accuracies = []
test_accuracies = []

# Support Vector Machine (SVM)
# hyperparameters = {
    # 'learning_rate': 0.01,
    # 'max_depth': 10,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.3,
    # 'lambda': 0.6,
    # 'alpha': 0.3,
    # 'gamma': 0.0,
    # 'base_score': 0.5,
    # 'min_child_weight': 5,
    # 'base_score': y_train.mean()
    # }

# hyperparameters = {
     # 'learning_rate': 0.1,
     # 'max_depth': 4,
     # 'lambda': 0.6,
     # 'alpha': 0.3,
     # 'gamma': 0.0,
     # 'base_score': 0.5,
     # 'min_child_weight': 5,
     # 'base_score': y_train.mean()
 # }

# num_boost_round = 300

# xgb_model = XGBClassifier('binary:logistic', learning_rate = 0.1, max_depth = 10, n_estimators = 1500, subsample = 0.5, colsample_bytree = 0.3,
#                         use_label_encoder = False, eval_metric = 'logloss')
xgb_model = XGBClassifier('binary:logistic', learning_rate = 0.2, max_depth = 9, n_estimators = 300, subsample = 0.8, colsample_bytree = 0.8)
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

# Get raw outputs (logits)
# train_logits = xgb_model.base.predict(X_train)
# test_logits = xgb_model.base.predict(X_test)

# Convert to probabilities using sigmoid
# train_probs = 1 / (1 + np.exp(-train_logits))
# test_probs = 1 / (1 + np.exp(-test_logits))

# Convert probabilities to binary predictions (0 or 1)
# train_predict_xgb = (train_probs >= 0.5).astype(int)
# test_predict_xgb = (test_probs >= 0.5).astype(int)

train_predict_xgb = xgb_model.predict(X_train)
test_predict_xgb = xgb_model.predict(X_test)

cm_train_xgb = confusion_matrix(y_train, train_predict_xgb)
cm_test_xgb = confusion_matrix(y_test, test_predict_xgb)

xgb_train_acc = accuracy_score(y_train, train_predict_xgb)
xgb_test_acc = accuracy_score(y_test, test_predict_xgb)
    
train_accuracies.append(xgb_train_acc)
test_accuracies.append(xgb_test_acc)

report_train = classification_report(y_train, train_predict_xgb, target_names = levels)
report_test = classification_report(y_test, test_predict_xgb, target_names = levels)

# Print results
print(f"XGBoost Train Accuracy: {xgb_train_acc * 100:.2f}%")
print(f"XGBoost Test Accuracy: {xgb_test_acc * 100:.2f}%")

print("Classification Report (Training):\n", report_train)
print("Classification Report (Testing):\n", report_test)

# print("Estimators:", colsample_bytree)
# print("Train accuracies:", train_accuracies)
# print("Test accuracies:", test_accuracies)

#  --- Visualize the confusion matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=levels, yticklabels=levels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 3. Plot the Learning Curve (Overfitting Check)
results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.xlabel('Number of Iterations')
plt.title('XGBoost Learning Curve (Overfitting Analysis)')
plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(colsample_bytree, train_accuracies, marker='o', label='Training Accuracy')
# plt.plot(colsample_bytree, test_accuracies, marker='s', label='Testing Accuracy')

# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# plt.xlabel('Colsample Bytree')
# plt.ylabel('Accuracy')
# plt.title('Training  vs Test Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save XGBoost
with open('models/xgb_model.pkl', 'wb') as f:
     pickle.dump(xgb_model, f)