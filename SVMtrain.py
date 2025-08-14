import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

# 1️. Load & Prepare Dataset (Binary Classification)
cancer = datasets.load_breast_cancer()
X = cancer.data[:, :2]  # taking only 2 features for 2D visualization
y = cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2. Train SVM with Linear Kernel
svc_linear = SVC(kernel='linear', C=1.0, random_state=42)
svc_linear.fit(X_train, y_train)

# 3️. Train SVM with RBF Kernel
svc_rbf = SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=42)
svc_rbf.fit(X_train, y_train)

# 4️. Visualize Decision Boundaries
plt.figure(figsize=(12, 5))

# Linear Kernel Plot
plt.subplot(1, 2, 1)
plot_decision_regions(X_train, y_train, clf=svc_linear, legend=2)
plt.title('SVM - Linear Kernel (Train Data)')

# RBF Kernel Plot
plt.subplot(1, 2, 2)
plot_decision_regions(X_train, y_train, clf=svc_rbf, legend=2)
plt.title('SVM - RBF Kernel (Train Data)')

plt.show()

# 5️. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters Found:", grid.best_params_)
print("Best Cross-validation Accuracy:", grid.best_score_)

# 6️. Cross-validation Score for Final Model
final_model = grid.best_estimator_
cv_scores = cross_val_score(final_model, X, y, cv=5)
print("Final Model Cross-validation Accuracy:", np.mean(cv_scores))

# 7️. Test Set Accuracy
print("Test Set Accuracy:", final_model.score(X_test, y_test))
