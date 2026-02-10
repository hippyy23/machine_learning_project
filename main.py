import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


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
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

# Test the model
y_pred_knn = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of KNN: {accuracy: .4f} ({accuracy*100: .2f}%)")

# Model 2: SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy of SVM: {accuracy: .4f} ({accuracy * 100: .2f}%)")

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
svm_scores = cross_val_score(svm, X_scaled_full, y, cv=5)

print(f"Cross validation scores for SVM: {svm_scores}")
print(f"Average SVM accuracy: {np.mean(svm_scores) * 100: .2f}%")

dt_scores = cross_val_score(dt, X, y, cv=5)
print(f"Decision Tree average accuracy: {np.mean(dt_scores) * 100: .2f}%")