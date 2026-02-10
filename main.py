import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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