import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)

print(classification_report(y_test, y_pred))

### Stratified ###
oversampling.perform_stratified_cv(log_reg_model, X, y)


### Oversampling ###
log_regressor = oversampling.perform_random_oversampling(log_reg_model,X_train,y_train,X_test,y_test)

proba = log_regressor.predict_proba(X_test)

proba_positive_class = proba[:, 1]

percentage = proba_positive_class * 100

### Smote ###
oversampling.perform_smote_oversampling(log_reg_model,X_train,X_test,y_train,y_test)




