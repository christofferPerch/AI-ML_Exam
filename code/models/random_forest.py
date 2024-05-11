import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import importlib
import oversampling
importlib.reload(oversampling)


#df = pd.read_csv("../data/heart_2022_transformed_no_outliers.csv")
df = pd.read_csv("../data/heart_2022_transformed_with_outliers.csv")

# Define features and target.
X = df.drop("HadHeartDisease", axis=1)
y = df["HadHeartDisease"]

# Splitting the dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### Without resampling ###
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

print(classification_report(y_test, y_pred_rf))

### Stratified ###

oversampling.perform_stratified_cv(rf_classifier, X, y)

### Random oversampling ###
oversampling.perform_random_oversampling(rf_classifier,X_train,y_train,X_test,y_test)

### Smote ###
oversampling.perform_smote_oversampling(rf_classifier,X_train,X_test,y_train,y_test)
