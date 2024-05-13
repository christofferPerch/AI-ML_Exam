import pandas as pd
import importlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import code.models.features.oversampling as oversampling
from sklearn.metrics import classification_report

# Reloads the oversampling module to ensure the latest version is used:
importlib.reload(oversampling)

# Loading the dataset:
df = pd.read_csv("../../data/processed/heart_2022_transformed.csv")
df_gs = df.copy()

# Defining the feature set by excluding the target variable, and also specifies the target variable:
X = df.drop("HadHeartDisease", axis=1)
y = df["HadHeartDisease"]

# Splitting the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model and parameter grid for optimization:
model = LogisticRegression()
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["newton-cg", "lbfgs", "liblinear"],
    "penalty": ["l2"],
}

# Setup GridSearchCV to find the best model parameters:
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
)

# Fit GridSearchCV on the training data:
grid_search.fit(X_train, y_train)

# Evaluate the best model from GridSearchCV:
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Prints classification report for the best model:
print(classification_report(y_test, y_pred))

## STRATIFIED SAMPLING ##
# Apply stratified cross-validation using the best model:
oversampling.perform_stratified_cv(best_model, X, y)

## RANDOM OVERSAMPLING ##
# Apply Random Oversampling using the best model and evaluate:
log_regressor = oversampling.perform_random_oversampling(
    best_model, X_train, y_train, X_test, y_test
)

# Getting the predicted probabilities for the test data:
proba = log_regressor.predict_proba(X_test)

# Isolating the probabilities for the positive class:
proba_positive_class = proba[:, 1]

# Converting the probabilities to percentages:
percentage = proba_positive_class * 100

## SMOTE ##
# Apply SMOTE oversampling using the best model and evaluate it:
oversampling.perform_smote_oversampling(best_model, X_train, X_test, y_train, y_test)

""" All results can be found inside the directory "./results/model_results.py". """
