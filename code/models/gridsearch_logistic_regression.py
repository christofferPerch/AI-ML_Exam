import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import importlib
import oversampling
importlib.reload(oversampling)

# Load data
df = pd.read_csv("../data/heart_2022_transformed_with_outliers.csv")

# Define features and target
X = df.drop("HadHeartDisease", axis=1)
y = df["HadHeartDisease"]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and parameter grid
model = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty': ['l2']
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Continue with your oversampling techniques as before
oversampling.perform_stratified_cv(best_model, X, y)
log_regressor = oversampling.perform_random_oversampling(best_model, X_train, y_train, X_test, y_test)
proba = log_regressor.predict_proba(X_test)
proba_positive_class = proba[:, 1]
percentage = proba_positive_class * 100
oversampling.perform_smote_oversampling(best_model, X_train, X_test, y_train, y_test)
