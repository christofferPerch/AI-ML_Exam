import os
import pandas as pd

# from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from pymongo import MongoClient
from load_data_from_mssql import load_data_from_sql
from pipelinetest import transform_data
from dotenv import load_dotenv

load_dotenv()
mongodb_url = os.getenv("MONGODB_URL")

df = load_data_from_sql()


def get_next_version(collection):
    # Retrieve the latest version of the model
    latest_model = collection.find_one(
        {"model_name": "logistic_regression"}, sort=[("version", -1)]
    )
    if latest_model:
        # Increment the version by 0.1
        latest_version = latest_model["version"]
        next_version = round(latest_version + 0.1, 1)
    else:
        # Start with version 1.0 if no model exists
        next_version = 1.0
    return next_version


def train_and_save_model():
    # Load data
    df = load_data_from_sql()

    # Transform data
    df_transformed = transform_data(df)

    # Define feature set and target variable
    X = df_transformed.drop("HadHeartDisease", axis=1)
    y = df_transformed["HadHeartDisease"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression model
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)

    # Save model to MongoDB
    client = MongoClient(mongodb_url)
    db = client["heart_disease"]
    collection = db["machineLearningModels"]

    # Get the next version number
    next_version = get_next_version(collection)

    # Save the model with versioning
    model_data = {
        "model_name": "logistic_regression",
        "version": next_version,
        "model": pickle.dumps(log_reg_model),
    }
    collection.insert_one(model_data)

    print(f"Model version {next_version} saved to MongoDB")


# Usage
if __name__ == "__main__":
    train_and_save_model()
