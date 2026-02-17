import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_model(model, y_true, y_pred):
    print(f"Evaluating {model} model...")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{cm}")

# Load data
df = pd.read_csv('data/heart-attack-data.csv')

print(f"Dataset Shape: {df.shape}")

print("\nMissing Values: ")
print(df.isnull().sum())

print("\nClass distribution: ")
print(df['target'].value_counts())

# Split data
X = df.drop('target', axis=1)
y = df['target']

# One-hot encoding
# defining which columns are categorical
# cp, thal, slope are categories
# sex, fbs, exang are already binary
categorical_cols = ['cp', 'restecg', 'slope', 'thal']

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"New shape after enconding: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scaled!")
print(f"First row of scaled training data: \n{X_train_scaled[0]}")

# Model 1: KNN
knn_params_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 21]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params_grid, cv=5)
knn_grid.fit(X_train_scaled, y_train)

print(f"Best parameters for KNN: {knn_grid.best_params_}")

# Use the best model
y_pred_knn = knn_grid.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of best KNN: {accuracy: .4f} ({accuracy * 100: .2f}%)")

# Save the best model
knn = knn_grid.best_estimator_

# Model 2: SVM
svm_param_grid = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10, 100]}
svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5)
svm_grid.fit(X_train_scaled, y_train)

print(f"Best parameters for SVM: {svm_grid.best_params_}")

# Use the best model
y_pred_svm = svm_grid.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy of SVM: {accuracy: .4f} ({accuracy * 100: .2f}%)")

# Save the best model
svm = svm_grid.best_estimator_

# Model 3: Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of Decision Tree: {accuracy: .4f} ({accuracy * 100: .2f}%)")

evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("Decision Tree", y_test, y_pred_dt)

# Cross validation
X_scaled_full = scaler.fit_transform(X)

knn_scores = cross_val_score(knn, X_scaled_full, y, cv=5)
print(f"Cross validation scores for KNN: {knn_scores}")
print(f"Average KNN accuracy: {np.mean(knn_scores) * 100: .2f}%")

svm_scores = cross_val_score(svm, X_scaled_full, y, cv=5)
print(f"Cross validation scores for SVM: {svm_scores}")
print(f"Average SVM accuracy: {np.mean(svm_scores) * 100: .2f}%")

dt_scores = cross_val_score(dt, X, y, cv=5)
print(f"Decision Tree average accuracy: {np.mean(dt_scores) * 100: .2f}%")

knn_cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_knn),
                                    display_labels=["Healthy", "Heart Attack"])

svm_cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_svm),
                                    display_labels=["Healthy", "Heart Attack"])

knn_cm_display.plot(cmap='Greens', colorbar=False)
plt.title("Confusion matrix for KNN model")
plt.show()

svm_cm_display.plot(cmap='Greens', colorbar=False)
plt.title("Confusion matrix for SVM model")
plt.show()