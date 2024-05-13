from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pipeline 

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    
    # Transform data
    transformed_data = pipeline.transform_data(data)
    
    # Load the trained model from the file
    with open("../models/saved_models/model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Make prediction
    prediction = model.predict_proba(transformed_data)
    percentage = prediction[:, 1] * 100
    
    # Return the prediction as JSON
    return jsonify({'prediction': f"{percentage.item():.2f}% probability of having the condition"})

if __name__ == '__main__':
    app.run(debug=True)