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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
from DecisionTree import DecisionTree

data = pd.read_csv("PhishingData.csv")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=41)

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


# Experiment 1
rf_model = RandomForest()
# Experiment 2
# rf_model = RandomForest(n_estimators = 300, max_depth = None, min_samples_split = 10, min_samples_leaf = 2, max_features = 'sqrt', max_leaf_nodes=2)

# Experiment 3
# rf_model = RandomForest(n_estimators = 300, max_depth = None, min_samples_split = 5, min_samples_leaf = 3, max_features = 'sqrt', max_leaf_nodes=4)


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

# K-nearest neighbour
def knn_accuracy(preds, y_test):
     return 100 * (preds == y_test).mean()


metric = 'euclidean'
knn_model = KNN(k=10, metric=metric)


knn_model.fit(X_train, y_train)
knn_predict = knn_model.predict(X_test)
# print("KNN Accuracy: ", knn_accuracy(preds, y_test), '%')
# print(f'Metric: {metric}, accuracy: {knn_accuracy(preds, y_test):.3f} %')

# Support Vector Machine (SVM)
svm_model = SVM(kernel='rbf', C=10.0, gamma=1.0, degree=3, coef0=0.0)
svm_model.fit(X_train, y_train)
predict_svm = svm_model.predict(X_test)
scores = cross_val_score(svm_model, X, y, cv=5)
# print("SVM accuracy:", accuracy(y_test, predict_svm), '%')

#  XGBoost

#  Hyperparameters for the XGBoost model
# # hyperparameters = {
#      'learning_rate': 0.3,
#      'max_depth': 6,
#      'subsample': 0.8,
#      'colsample_bytree': 0.8,
#      'lambda': 1.0,
#      'alpha': 1.0,
#      'gamma': 0.1,
#      'base_score': 0.5
# # }

hyperparameters = {
      'learning_rate': 0.1,
     'max_depth': 4,
     'subsample': 0.7,
     'colsample_bytree': 0.7,
     'lambda': 0.1,
     'alpha': 0.0,
     'gamma': 0.0,
     'base_score': 0.5,
     'min_child_weight': 0.1,
     'base_score': y_train.mean()
 }

num_boost_round = 300

xgb_model = XGBoostSigmoid(hyperparameters, seed=42)
xgb_model.train(X_train, y_train, num_boost_round)

# Get raw outputs (logits)
logits = xgb_model.base.predict(X_test)

# Convert to probabilities using sigmoid
probs = 1 / (1 + np.exp(-logits))

# Convert probabilities to binary predictions (0 or 1)
xgb_predict = (probs >= 0.5).astype(int)

# Evaluate accuracy
# xgb_acc = (xgb_predict == y_test).mean()

dt_model = DecisionTree(2,2)
dt_model.fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)
# accuracy = np.mean(xgb_predict == y_test)
# print("XGboost accuracy:", 100 * accuracy, '%')

# Calculating acccuracy for ensemble model
# pred_matrix = np.vstack([rf_predict, knn_predict, predict_svm, xgb_predict])
# final_preds = np.round(np.mean(pred_matrix, axis=0)).astype(int)

cm_train_rf = confusion_matrix(y_test, rf_train_predict)
cm_test_rf = confusion_matrix(y_test, rf_test_predict)
# cm_knn = confusion_matrix(y_test, knn_predict)
# cm_svm = confusion_matrix(y_test, predict_svm)
# cm_xgb = confusion_matrix(y_test, xgb_predict)
# cm_dt = confusion_matrix(y_test, dt_predict)

rf_train_acc = accuracy_score(y_train, rf_train_predict)
rf_test_acc = accuracy_score(y_test, rf_test_predict)
# knn_acc = accuracy_score(y_test, knn_predict)
# svm_acc = accuracy_score(y_test, predict_svm)
# xgb_acc = accuracy_score(y_test, xgb_predict)
# dt_acc = accuracy_score(y_test, dt_predict)

# report = classification_report(y_test, final_preds)

# Print results
print(f"Random Forest Train Accuracy: {rf_train_acc * 100:.2f}%")
print(f"Random Forest Test Accuracy: {rf_test_acc * 100:.2f}%")
# print(f"KNN Accuracy: {knn_acc * 100:.2f}%")
# print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
# print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")
# print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
# print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")

# print("Classification Report:\n", report)
# print("Confusion Matrix:\n", cm)

#  --- [4] Visualize the confusion matrix ---
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

# Save the models after training so that they can be used in the website later

# Save Random Forest
with open('models/rf_model.pkl', 'wb') as f:
     pickle.dump(rf_model, f)

# Save KNN
# with open('models/knn_model.pkl', 'wb') as f:
     pickle.dump(knn_model, f)

# Save SVM
# with open('models/svm_model.pkl', 'wb') as f:
     pickle.dump(svm_model, f)

# Save XGBoost
with open('models/xgb_model.pkl', 'wb') as f:
     pickle.dump(xgb_model, f)

with open('models/dt_model.pkl', 'wb') as f:
     pickle.dump(dt_model, f)

with open('models/scaler.pkl', 'wb') as f:
     pickle.dump(scaler, f)


