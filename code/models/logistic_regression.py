import pandas as pd
import importlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import code.models.features.oversampling as oversampling
from sklearn.metrics import classification_report

# Reloads the oversampling module to ensure the latest version is used:
importlib.reload(oversampling)

# Loading the dataset:
df = pd.read_csv("../../data/processed/heart_2022_transformed.csv")
df_lr = df.copy()


# Defining the feature set by excluding the target variable, and also specifies the target variable:
X = df.drop("HadHeartDisease", axis=1)
y = df["HadHeartDisease"]

# Splitting the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## WITHOUT RESAMPLING ##
# Here we initialize the Logistic Regression model:
log_reg_model = LogisticRegression()

# We then train the LR model on the training set:
log_reg_model.fit(X_train, y_train)

# Predicts the text set outcomes:
y_pred = log_reg_model.predict(X_test)

# Prints classification report for model evaluation:
print(classification_report(y_test, y_pred))

## STRATIFIED SAMPLING ##
# We perform stratified cross-validation:
oversampling.perform_stratified_cv(log_reg_model, X, y)


## RANDOM OVERSAMPLING ##
# Here we apply random oversampling to the training set and evaluates it:
log_regressor = oversampling.perform_random_oversampling(
    log_reg_model, X_train, y_train, X_test, y_test
)

# Getting the predicted probabilities for the test data:
proba = log_regressor.predict_proba(X_test)

# Isolating the probabilities for the positive class:
proba_positive_class = proba[:, 1]

percentage = proba_positive_class * 100

## SMOTE ##
# Here we apply SMOTE for oversampling and evaluates:
oversampling.perform_smote_oversampling(log_reg_model, X_train, X_test, y_train, y_test)


""" All results can be found inside the directory "./results/model_results.py". """
