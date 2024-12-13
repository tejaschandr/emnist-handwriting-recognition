import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # For multi-class classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load your DataFrame (replace this with your data loading code)
# Example: df = pd.read_csv("emnist_data.csv")
df = ...  # Assuming the DataFrame is already loaded

# Separate features (X) and labels (y)
X = df.iloc[:, :-1]  # All columns except the last one (pixel data)
y = df.iloc[:, -1]   # Last column (label)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Linear Regression (using Logistic Regression for classification)
print("Running Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_preds)
print(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")

# 2. Random Forest
print("Running Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# 3. Quadratic Discriminant Analysis (QDA)
print("Running QDA...")
qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)
qda_preds = qda_clf.predict(X_test)
qda_acc = accuracy_score(y_test, qda_preds)
print(f"QDA Accuracy: {qda_acc:.4f}")

# 4. PCA + Methods
print("Running PCA...")
n_components = 50  # Adjust the number of components based on your dataset
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Logistic Regression with PCA
print("Running Logistic Regression with PCA...")
log_reg_pca = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
log_reg_pca.fit(X_train_pca, y_train)
log_reg_pca_preds = log_reg_pca.predict(X_test_pca)
log_reg_pca_acc = accuracy_score(y_test, log_reg_pca_preds)
print(f"Logistic Regression with PCA Accuracy: {log_reg_pca_acc:.4f}")

# Random Forest with PCA
print("Running Random Forest with PCA...")
rf_pca_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca_clf.fit(X_train_pca, y_train)
rf_pca_preds = rf_pca_clf.predict(X_test_pca)
rf_pca_acc = accuracy_score(y_test, rf_pca_preds)
print(f"Random Forest with PCA Accuracy: {rf_pca_acc:.4f}")

# QDA with PCA
print("Running QDA with PCA...")
qda_pca_clf = QuadraticDiscriminantAnalysis()
qda_pca_clf.fit(X_train_pca, y_train)
qda_pca_preds = qda_pca_clf.predict(X_test_pca)
qda_pca_acc = accuracy_score(y_test, qda_pca_preds)
print(f"QDA with PCA Accuracy: {qda_pca_acc:.4f}")
