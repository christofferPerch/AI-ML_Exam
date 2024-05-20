from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pipeline
import chatbot.chatbot_chat as chatbot_chat
import chatbot.chatbot_training as chatbot_training
from pymongo import MongoClient
from retrain.retrain_logistic_regression import (
    train_and_save_model as train_and_save_model,
)
from retrain.retrain_random_forest import (
    train_and_save_model as train_and_save_model_rf,
)
from retrain.retrain_tensorflow import train_and_save_model as train_and_save_model_tf


app = Flask(__name__)


@app.route("/genai_embed", methods=["POST"])
def genai_embed():
    chatbot_training.train_chatbot()
    return jsonify({"message": "Embedding process completed successfully."})


@app.route("/genai_chat", methods=["POST"])
def genai_chat():
    data = request.get_json()
    question = data["message"]
    response = chatbot_chat.chat(question)
    return jsonify({"response": response})


def get_latest_model():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["heart_disease"]
    collection = db["machineLearningModels"]

    # Retrieve the latest model by sorting by version in descending order
    latest_model_entry = collection.find_one(
        {"model_name": "logistic_regression"}, sort=[("version", -1)]
    )

    if latest_model_entry:
        model = pickle.loads(latest_model_entry["model"])
        return model
    else:
        raise ValueError("No model found in the database.")


@app.route("/predict_lr", methods=["POST"])
def predict_logistic_regression():
    # Get data from request
    data = request.get_json()

    # Transform data
    transformed_data = pipeline.transform_data(data)

    # Load the latest model from MongoDB
    model = get_latest_model()

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

    # Transform data
    transformed_data = pipeline.transform_data(data)

    # Load the trained model from the file
    with open("../models/saved_models/model_random_forest.pkl", "rb") as file:
        model = pickle.load(file)

    # Make prediction
    prediction = model.predict_proba(transformed_data)
    percentage = prediction[:, 1] * 100

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

    # Transform data
    transformed_data = pipeline.transform_data(data)

    # Load the trained model from the file
    model = load_model("../models/saved_models/model_tensorflow.keras")

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
