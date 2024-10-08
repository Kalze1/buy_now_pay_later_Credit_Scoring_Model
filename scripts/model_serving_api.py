import pickle
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Step 1: Load the saved model (replace with actual saved model path)
model_file_name = f'../data/credit_scoring_logistic_model_2024-10-07.pkl'

with open(model_file_name, 'rb') as model_file:
    model = pickle.load(model_file)

# Step 2: Define preprocessing function
def preprocess_input(data):
    """
    Preprocesses the input data to match the model's expected format.
    This function will be called before making predictions.
    """
    try:
        # Convert the incoming JSON data to a pandas DataFrame
        df = pd.DataFrame([data])
        
        # Example: You might need to handle missing or unexpected values here
        # df = df.fillna(0)  # Replace NaN values, for example

        return df
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

# Step 3: Define API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions using the trained model.
    The input is expected in JSON format with features for the model.
    """
    try:
        # Step 4: Get JSON data from the request
        input_data = request.get_json()

        # Log the input data for debugging
        logging.debug(f"Received input data: {input_data}")

        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)

        # Step 5: Make predictions using the loaded model
        prediction = model.predict(preprocessed_data)

        # Step 6: Format the prediction output
        prediction_output = {'prediction': prediction.tolist()}

        # Return the prediction as a JSON response
        return jsonify(prediction_output)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Step 7: Run the Flask app (for local testing)
if __name__ == '__main__':
    app.run(debug=True)
