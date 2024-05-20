import pandas as pd
from gridfs import GridFS

# from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import pickle
from pymongo import MongoClient

# import code.models.features.oversampling as oversampling
from sklearn.ensemble import RandomForestClassifier
from load_data_from_mssql import load_data_from_sql
from pipelinetest import transform_data
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler


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

        # Prints the classification report for the current fold:
        # print(f"\nClassification Report for Fold {i+1}:\n")
        # print(classification_report(y_test_fold, y_pred))
        return model_clone


def get_next_version(collection):
    # Retrieve the latest version of the model
    latest_model = collection.find_one(
        {"model_name": "random_forest"}, sort=[("version", -1)]
    )
    if latest_model:
        # Increment the version by 0.1
        latest_version = latest_model["version"]
        next_version = round(latest_version + 0.1, 1)
    else:
        # Start with version 1.0 if no model exists
        next_version = 1.0
    return next_version


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

    # print(classification_report(y_test, y_pred))
    return model_clone


def train_and_save_model():
    # Load data
    df = load_data_from_sql()

    # Transform data
    df_transformed = transform_data(df)

    # Define feature set and target variable
    X = df_transformed.drop("HadHeartDisease", axis=1)
    y = df_transformed["HadHeartDisease"]

    rf_classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )

    random_forest = perform_stratified_cv(rf_classifier, X, y)

    # Save model to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["heart_disease"]
    collection = db["machineLearningModels"]
    fs = GridFS(db)

    # Get the next version number
    next_version = get_next_version(collection)

    # use gridfs to store the model as it is bigger than BSON-document size limit of 16MB
    model_data = pickle.dumps(random_forest)
    grid_out = fs.put(model_data, filename=f"random_forest_{next_version}.pkl")

    # Save the model with versioning
    metadata = {
        "model_name": "random_forest",
        "version": next_version,
        "model_id": grid_out,  # This is the _id of the file in GridFS
    }
    collection.insert_one(metadata)

    print(f"Model version {next_version} saved to MongoDB")


# Usage
if __name__ == "__main__":
    train_and_save_model()
