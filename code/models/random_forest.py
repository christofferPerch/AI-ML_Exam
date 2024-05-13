import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import importlib
import code.models.features.oversampling as oversampling

# Reloads the oversampling module to ensure the latest version is used:
importlib.reload(oversampling)

# Loading the dataset:
df = pd.read_csv("../../data/processed/heart_2022_transformed.csv")
df_rf = df.copy()

# Defining the feature set by excluding the target variable, and also specifies the target variable:
X = df_rf.drop("HadHeartDisease", axis=1)
y = df_rf["HadHeartDisease"]

# Splitting the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## WITHOUT RESAMPLING ##
# Here we initialize the RandomForestClassifier with balanced class weights:
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)

# We then train the RF model on the training set:
rf_classifier.fit(X_train, y_train)

# Predicts the test set outcomes:
y_pred_rf = rf_classifier.predict(X_test)

# Prints classification report for model evaluation:
print(classification_report(y_test, y_pred_rf))


## STRATIFIED SAMPLING ##
# We evaluate the classifier using stratified cross-validation:
oversampling.perform_stratified_cv(rf_classifier, X, y)

## RANDOM OVERSAMPLING ##
# Here we apply random oversampling to the training set and evaluates it:
oversampling.perform_random_oversampling(
    rf_classifier, X_train, y_train, X_test, y_test
)

## SMOTE ##
# Here we apply SMOTE for oversampling and evaluates:
oversampling.perform_smote_oversampling(rf_classifier, X_train, X_test, y_train, y_test)


""" All results can be found inside the directory "./results/model_results.py". """
