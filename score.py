import json
import joblib
import numpy as np
import os

# The init() function runs once when the container starts.
def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created by Azure ML
    # It points to the folder where your registered model is downloaded.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'diabetes_model.pkl')
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")

# The run() function runs every time someone pings your API.
def run(raw_data):
    try:
        # Azure ML passes the JSON payload as a string
        json_data = json.loads(raw_data)
        
        # Extract the list of features (assuming the payload has a "data" key)
        # Example input: {"data": [[1, 120, 70, 25.5, 30]]}
        input_data = np.array(json_data['data'])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return the result as a JSON-serializable dictionary
        return {"diabetic": bool(prediction[0])}
    
    except Exception as e:
        error_msg = str(e)
        return {"error": error_msg}