import pandas as pd

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