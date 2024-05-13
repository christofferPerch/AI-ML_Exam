import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


"""
We have decided to use three technique of sampling, Stratified, Random Oversampling and SMOTE.

These three techniques helps us address imbalances in the dataset, particularly since our use case
of people with and without a heart disease condition are significantly unevenly distributed.

If we didn't use resampling techniques, these imbalances could potentially lead to biased models,
that perform well on the majority class, which would be individuals without a heart disease, and
poorly on the minority class which would be the individuals with a heart disease or condition.

By using Oversampling it helps us adjust the dataset to have a more balanced distribution,
which helps improve our model's ability to learn from and make predictions more effectively.
"""


# Stratified Sampling method:
def perform_stratified_cv(model, X, y, n_splits=2, sample_frac=1.0):

    # Initializing stratified k-folds:
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Iterates over each fold, and splits features and target unti training and test sets:
    for i, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Using a fraction of the training set if sample_frac is less than 1.0:
        if sample_frac < 1.0:
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train_fold, y_train_fold, test_size=1 - sample_frac, random_state=42
            )
        else:
            # Use the full training data if sample_frac is 1.0:
            X_train_sample, y_train_sample = X_train_fold, y_train_fold

        # Clones the model to avoid interference between folds:
        model_clone = clone(model)
        # Here it clones the model on the training subset, and predicts the target for the test fold:
        model_clone.fit(X_train_sample, y_train_sample)
        y_pred = model_clone.predict(X_test_fold)

        # Prints the classification report for the current fold:
        print(f"\nClassification Report for Fold {i+1}:\n")
        print(classification_report(y_test_fold, y_pred))


# Random Oversampling method:
def perform_random_oversampling(
    model, X_train, y_train, X_test, y_test, random_state=42
):
    # Initializing the random oversampler:
    rd_oversampler = RandomOverSampler(random_state=random_state)

    # Applies random oversampling to the training data:
    X_train_oversampled, y_train_oversampled = rd_oversampler.fit_resample(
        X_train, y_train
    )

    # Clones the model to avoid interference:
    model_clone = clone(model)
    # Trains the model on the oversampled training data:
    model_clone.fit(X_train_oversampled, y_train_oversampled)

    # Predicts the target for the test data:
    y_pred = model_clone.predict(X_test)

    print(classification_report(y_test, y_pred))
    return model_clone


# SMOTE method:
def perform_smote_oversampling(
    model, X_train, X_test, y_train, y_test, random_state=42
):

    # Initializing SMOTE for synthetic minority oversampling
    smote = SMOTE(random_state=random_state)

    # Applies SMOTE to the training data:
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Clones the model to avoid interference:
    model_clone = clone(model)
    # Trains the model on the SMOTE processed data:
    model_clone.fit(X_train_smote, y_train_smote)

    # Predicts the target for the test data:
    y_pred = model_clone.predict(X_test)

    print(classification_report(y_test, y_pred))
