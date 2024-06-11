import os
import math
import tensorflow as tf
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler

# from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

# import code.models.features.oversampling as oversampling
from ..load_data_from_mssql import load_data_from_sql
from ...data_wrangling.data_transformation import transform_data


load_dotenv()
mongodb_url = os.getenv("MONGODB_URL")


def get_next_version(collection):
    # Retrieve the latest version of the model
    latest_model = collection.find_one(
        {"model_name": "tensorflow"}, sort=[("version", -1)]
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
    # Here we make a custom learning rate scheduler function:
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr * math.exp(-0.1)
        else:
            return lr * math.exp(-0.2)  # More aggressive reduction after 10 epochs.

    # Initializing the learning rate scheduler:
    lr_scheduler = LearningRateScheduler(scheduler)

    # Loading the dataset:
    df = load_data_from_sql()

    df = transform_data(df)
    df_nn = df.copy()

    # Split data into features and target variables:
    X = df_nn.drop("HadHeartDisease", axis=1)
    y = df_nn["HadHeartDisease"]

    # Splitting the dataset into training and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Applying Random Oversampling to the training data:
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Defining the structure of the neural network:
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(X_train_resampled.shape[1],)),
            BatchNormalization(),
            Dropout(0.1),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Compiling the model with Adam optimizer and setting the loss and metrics:
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", Recall(), Precision()],
    )

    # Initializing callbacks for reducing learning rate and early stopping:
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    # Training the model with specified callbacks and hyperparameters:
    callbacks = [reduce_lr, early_stopping, lr_scheduler]
    history = model.fit(
        X_train_resampled,
        y_train_resampled,
        epochs=5,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        batch_size=64,
        verbose=1,
    )

    # Save the TensorFlow model to a directory
    model.save("model_tensorflow.keras")

    client = MongoClient(mongodb_url)
    db = client["heart_disease"]
    collection = db["machineLearningModels"]

    # Get the next version number
    next_version = get_next_version(collection)

    temp_file = "temp_model.keras"
    tf.keras.models.save_model(model, temp_file)

    # Read the temporary file as binary data
    with open(temp_file, "rb") as f:
        binary_data = f.read()

    # Remove the temporary file
    os.remove(temp_file)

    # Save the model with versioning
    model_data = {
        "model_name": "tensorflow",
        "version": next_version,
        "model": binary_data,
    }
    collection.insert_one(model_data)

    print(f"Model version {next_version} saved to MongoDB")


# Usage
if __name__ == "__main__":
    train_and_save_model()
