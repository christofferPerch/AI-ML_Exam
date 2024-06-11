import os
from flask import Flask, request, jsonify
import pickle
import gridfs
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

from ..data_wrangling.data_transformation import transform_data
from .chatbot.chatbot_chat import chat
from .chatbot.chatbot_training import train_chatbot

from pymongo import MongoClient
from .retrain.retrain_logistic_regression import (
    train_and_save_model as train_and_save_model,
)
from .retrain.retrain_random_forest import (
    train_and_save_model as train_and_save_model_rf,
)
from .retrain.retrain_tensorflow import train_and_save_model as train_and_save_model_tf

load_dotenv()
mongodb_url = os.getenv("MONGODB_URL")

app = Flask(__name__)


@app.route("/genai_embed", methods=["POST"])
def genai_embed():
    train_chatbot()
    return jsonify({"message": "Embedding process completed successfully."})


@app.route("/genai_chat", methods=["POST"])
def genai_chat():
    data = request.get_json()
    question = data["message"]
    response = chat(question)
    return jsonify({"response": response})


def get_latest_model(model_name):
    client = MongoClient(mongodb_url)
    db = client["heart_disease"]
    collection = db["machineLearningModels"]

    # Retrieve the latest model by sorting by version in descending order
    latest_model_entry = collection.find_one(
        {"model_name": model_name}, sort=[("version", -1)]
    )

    if latest_model_entry:
        if model_name == "tensorflow":
            # Create a temporary file and write the model to it
            with open("temp.keras", "wb") as tmp:
                tmp.write(latest_model_entry["model"])

            # Load the model from the temporary file
            model = tf.keras.models.load_model("temp.keras")

            # Delete the temporary file
            os.remove("temp.keras")
        if model_name == "random_forest":
            fs = gridfs.GridFS(db)
            grid_out = fs.get(latest_model_entry["model_id"])
            with open("temp.pkl", "wb") as tmp:
                tmp.write(grid_out.read())

            model = pickle.loads(open("temp.pkl", "rb").read())

            # Delete the temporary file
            os.remove("temp.pkl")
        if model_name == "logistic_regression":
            model = pickle.loads(latest_model_entry["model"])
        return model
    else:
        raise ValueError("No model found in the database.")


@app.route("/predict_lr", methods=["POST"])
def predict_logistic_regression():
    # Get data from request
    data = request.get_json()

    # Convert dictionary to DataFrame
    try:
        df = pd.DataFrame(data, index=[0])  # Assuming data is for a single record
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Transform data
    transformed_data = transform_data(df, True)

    # Load the latest model from MongoDB
    model = get_latest_model("logistic_regression")

    # Make prediction
    prediction = model.predict_proba(transformed_data)
    percentage = prediction[:, 1] * 100

    # Return the prediction as JSON
    return jsonify(
        {
            "prediction": f"{percentage.item():.2f}% probability of getting a heart disease."
        }
    )


@app.route("/predict_rf", methods=["POST"])
def predict_random_forest():
    # Get data from request
    data = request.get_json()

    try:
        df = pd.DataFrame(data, index=[0])  # Assuming data is for a single record
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Transform data
    transformed_data = transform_data(df, True)

    # Load the trained model from the file
    model = get_latest_model("random_forest")

    # Make prediction
    prediction = model.predict_proba(transformed_data)
    percentage = prediction[:, 1] * 100

    print("PREDICTION", prediction)

    # Return the prediction as JSON
    return jsonify(
        {
            "prediction": f"{percentage.item():.2f}% probability of getting a heart disease."
        }
    )


@app.route("/predict_tf", methods=["POST"])
def predict_tensorflow():
    # Get data from request
    data = request.get_json()

    # Convert dictionary to DataFrame
    try:
        df = pd.DataFrame(data, index=[0])  # Assuming data is for a single record
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Transform data
    transformed_data = transform_data(df, True)

    # Load the trained model from the file
    model = get_latest_model("tensorflow")

    # Make prediction
    prediction = model.predict(transformed_data)
    percentage = prediction * 100

    # Return the prediction as JSON
    return jsonify(
        {
            "prediction": f"{percentage.item():.2f}% probability of getting a heart disease."
        }
    )


@app.route("/retrain_lr", methods=["GET"])
def retrain_model_lr():
    try:
        train_and_save_model()
        return jsonify({"message": "Model retrained and saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrain_rf", methods=["GET"])
def retrain_model_rf():
    try:
        train_and_save_model_rf()
        return jsonify({"message": "Model retrained and saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/retrain_tf", methods=["GET"])
def retrain_model_tf():
    try:
        train_and_save_model_tf()
        return jsonify({"message": "Model retrained and saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
