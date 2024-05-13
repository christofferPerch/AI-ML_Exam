from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pipeline 

app = Flask(__name__)


@app.route('/predict_lr', methods=['POST'])
def predict_logistic_regression():
    # Get data from request
    data = request.get_json()
    
    # Transform data
    transformed_data = pipeline.transform_data(data)
    
    # Load the trained model from the file
    with open("../models/saved_models/model_logistic_regression.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Make prediction
    prediction = model.predict_proba(transformed_data)
    percentage = prediction[:, 1] * 100
    
    # Return the prediction as JSON
    return jsonify({'prediction': f"{percentage.item():.2f}% probability of having the condition"})

@app.route('/predict_rf', methods=['POST'])
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
    return jsonify({'prediction': f"{percentage.item():.2f}% probability of having the condition"})

@app.route('/predict_tf', methods=['POST'])
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
    return jsonify({'prediction': f"{percentage.item():.2f}% probability of having the condition"})



if __name__ == '__main__':
    app.run(debug=True)