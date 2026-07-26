
# Import data manipulation libraries
import numpy as np
import pandas as pd

# For serialization
import joblib

# Flask API
from flask import Flask, request, jsonify

# Import logging
import logging
import sys

# Initialize the Flask app with a name
superkart_api = Flask("superkart_sales_app")

# Debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Module name: {__name__}")
logger.info(f"Flask app name: {superkart_api.name}")
logger.info(f"Root path: {superkart_api.root_path}")

# Load the trained churn prediction model
model = joblib.load("forecast_superkart_sales_model.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a welcome message and some help on the endpoints supported.
    """
    logger.info("Home endpoint accessed")

    html = """
      <!DOCTYPE html>
      <html>
      <head>
        <title>SuperKart Sales API</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background-color: #f4f4f4;
          }
          h1 {
            color: #333;
            font-size: 3em;
          }
          p {
            color: #666;
            font-size: 1.5em;
            margin-top: 20px;
          }
        </style>
      </head>
      <body>
        <h1>Welcome to SuperKart Sales Prediction API.</h1>
        <p>To obtain sales predictions, please send a POST request to `/v1/sales`.</p>
      </body>
      </html>
    """
    return html

# Define an endpoint to predict a product sale
@superkart_api.post('/v1/sales')
def predict_sales():
    """
    This hanldes the POST requests to the endpoint /v1/sales
    It gets JSON data from the request and returns predicted sale

    """
    try:
      # Get the JSON data from the request body
      data = request.get_json()

      # Extract relevant product features from the input data.
      sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category'],
        'Product_Id_char': data['Product_Id_char']
      }

      # Convert the extracted data into a DataFrame
      input_data = pd.DataFrame([sample])
      logger.debug(f"Input DataFrame:\n{input_data}")

      # Make a prediction using the trained model
      prediction = model.predict(input_data).tolist()[0]

      # Return the prediction as a JSON response
      return jsonify({'Sales': prediction})

    except KeyError as e:
      return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
      return jsonify({'error': f'Prediction failed: {str(e)}' }), 500

# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/salesbatch')
def predict_sales_batch():
    try:
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the CSV file into a Pandas DataFrame
        input_data = pd.read_csv(file)

        # Ensure the expected feature columns exist before predicting.
        expected_columns = [
            'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_MRP', 'Store_Size', 'Store_Location_City_Type', 'Store_Type',
            'Product_Id_char', 'Store_Age_Years', 'Product_Type_Category'
        ]
        missing_columns = [col for col in expected_columns if col not in input_data.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {missing_columns}'}), 400

        # Make predictions for all properties in the DataFrame (get sales predictions)
        predicted_sales = model.predict(input_data[expected_columns]).tolist()

        # Convert predictions to a user-friendly numeric format.
        predicted_sale = [round(float(sales), 2) for sales in predicted_sales]

        # Return predictions as a list of objects. If an id column exists, include it.
        if 'id' in input_data.columns:
            output = [
                {'id': row_id, 'Sales': sales}
                for row_id, sales in zip(input_data['id'].tolist(), predicted_sale)
            ]
        else:
            output = [{'Sales': sales} for sales in predicted_sale]

        return jsonify(output)
    except Exception as e:
        logger.exception('Batch prediction failed')
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
