import pickle
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load the saved model (replace with actual saved model path)
model_file_name = f'../data/credit_scoring_logistic_model_2024-10-07.pkl'

with open(model_file_name, 'rb') as model_file:
    model = pickle.load(model_file)

# Step 2: Define preprocessing function (if needed, adjust based on your model's expected input)
def preprocess_input(data):
    """
    Preprocesses the input data to match the model's expected format.
    This function will be called before making predictions.
    """
    # Convert the incoming JSON data to a pandas DataFrame
    df = pd.DataFrame([data])
    
    # Example: Drop any unnecessary columns or do data transformations
    # For instance, if your model expects certain columns to be scaled, encoded, etc.
    # df = df.drop(columns=['unnecessary_column'])
    
    return df

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

        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)

        # Step 5: Make predictions using the loaded model
        prediction = model.predict(preprocessed_data)

        # Step 6: Format the prediction output
        prediction_output = {'prediction': prediction.tolist()}  # Convert to list for JSON serialization

        # Return the prediction as a JSON response
        return jsonify(prediction_output)

    except Exception as e:
        # Handle any errors that occur during processing
        return jsonify({'error': str(e)}), 400

# Step 7: Run the Flask app (for local testing)
if __name__ == '__main__':
    app.run(debug=True)
