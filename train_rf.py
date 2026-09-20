import numpy as np
import pandas as pd
from RandomForest import RandomForest
from KNN import KNN
from sklearn.tree import DecisionTreeClassifier
from SVM import SVM
from PCA import PCA
from XGBoostSigmoid import XGBoostSigmoid
import pickle
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# Random Forest Classifier
# rf_model = RandomForest(10,10,2)

# param_grid = {
    # 'n_estimators': 300,       # number of trees
    # 'max_depth': None,        # tree depth
    # 'min_samples_split': 10,       # minimum samples to split a node
    # 'min_samples_leaf': 2,         # minimum samples per leaf
    # 'max_features': 'sqrt' # number of features to consider at each split
# }

max_leaf_nodes = list(range(100, 1101, 100))
train_accuracies = []
test_accuracies = []

# Experiment 1
# rf_model = RandomForest(n_estimators = 100, max_depth = 20, min_samples_split = 8, min_samples_leaf = 1, max_features = 2, max_leaf_nodes = 1024)
rf_model = RandomForest(n_estimators = 100, max_depth = 20, min_samples_split = 5, min_samples_leaf = 1, max_features = 8, max_leaf_nodes = 1024)
# grid_search = GridSearchCV(
    # estimator=rf_model,
    # param_grid=param_grid,
    # cv=5,                # 5-fold cross-validation
    # scoring='accuracy',  # or 'f1', 'recall', etc.
    # verbose=2
# )

# grid_search.fit(X_train, y_train)
# best_rf = grid_search.best_estimator_

# print("Best Parameters:", grid_search.best_params_)
# print("Best Accuracy:", grid_search.best_score_)

rf_model.fit(X_train, y_train)

rf_train_predict = rf_model.predict(X_train)
rf_test_predict = rf_model.predict(X_test) # evaluate the model on the test data
# print("Random Forest accuracy:", accuracy(y_test, predictions), '%')

cm_train_rf = confusion_matrix(y_train, rf_train_predict)
cm_test_rf = confusion_matrix(y_test, rf_test_predict)

rf_train_acc = accuracy_score(y_train, rf_train_predict)
rf_test_acc = accuracy_score(y_test, rf_test_predict)

train_accuracies.append(rf_train_acc)
test_accuracies.append(rf_test_acc)

report_train = classification_report(y_train, rf_train_predict, target_names = levels)
report_test = classification_report(y_test, rf_test_predict, target_names = levels)

# Print results
print(f"Random Forest Train Accuracy: {rf_train_acc * 100:.2f}%")
print(f"Random Forest Test Accuracy: {rf_test_acc * 100:.2f}%")

print("Classification Report (Training):\n", report_train)
print("Classification Report (Testing):\n", report_test)

# print("Estimators:", max_leaf_nodes)
# print("Train accuracies:", train_accuracies)
# print("Test accuracies:", test_accuracies)

#  --- Visualize the confusion matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_rf, annot=True, fmt='d', cmap='Blues', xticklabels=levels, yticklabels=levels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(max_leaf_nodes, train_accuracies, marker='o', label='Training Accuracy')
# plt.plot(max_leaf_nodes, test_accuracies, marker='s', label='Testing Accuracy')

# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# plt.xlabel('Max Leaf Nodes')
# plt.ylabel('Accuracy')
# plt.title('Training  vs Test Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

#plt.figure(figsize=(8, 6))
#plt.plot(max_leaf_nodes, test_accuracies, marker='o', label='Test Accuracy')

#plt.xlabel('Max leaf nodes')
#plt.ylabel('Accuracy')
#plt.title('Test Accuracy')
#plt.legend()
#plt.grid(True)
#plt.show()

# Save Random Forest
with open('models/rf_model.pkl', 'wb') as f:
     pickle.dump(rf_model, f)


