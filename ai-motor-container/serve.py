from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib # Import joblib here, as load_pipeline and load_label_encoder will likely use it

# Assuming these imports exist and work:
from inference import predict_failure_type
from model_loader import load_pipeline, load_label_encoder

# Initialize Flask
app = Flask(__name__)

# --- Model/Encoder Loading with Robust Logging ---
pipeline = None
label_encoder = None

def initialize_models():
    global pipeline, label_encoder
    print("INFO: Starting model and label encoder initialization...")
    
    # Debug environment variables that Vertex AI sets
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    print(f"DEBUG: AIP_MODEL_DIR environment variable: {aip_model_dir}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Contents of current working directory: {os.listdir(os.getcwd())}")
    
    # If your models are in a 'models' subdirectory, ensure to list that too
    if 'models' in os.listdir(os.getcwd()):
        print(f"DEBUG: Contents of ./models/: {os.listdir(os.path.join(os.getcwd(), 'models'))}")

    try:
        # These functions (load_pipeline, load_label_encoder) should implement
        # their own robust file path handling and error logging internally,
        # as per our previous discussion.
        pipeline_obj = load_pipeline()
        label_encoder_obj = load_label_encoder()

        if pipeline_obj is None or label_encoder_obj is None:
            raise ValueError("One or both models/encoders failed to load from model_loader.py.")

        pipeline = pipeline_obj
        label_encoder = label_encoder_obj
        print("INFO: All models and encoders loaded successfully at startup.")
    except Exception as e:
        print(f"FATAL ERROR: Application failed to load models/encoders at startup. Exception: {e}")
        # Re-raise the exception to make sure Gunicorn worker crashes
        # and sends this error to stdout/stderr, which Cloud Logging captures.
        raise

# Call the initialization function when the app starts
try:
    initialize_models()
except Exception:
    # If model loading fails, we'll ensure the /ping endpoint reflects this.
    # The Gunicorn worker will likely crash and restart, but this helps.
    print("WARNING: Model initialization failed during Flask app startup. Endpoints may be unhealthy.")
    pipeline = None # Ensure they are None if loading failed
    label_encoder = None

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None or label_encoder is None:
        print("ERROR: Prediction requested but models/encoders are not loaded.")
        return jsonify({"error": "Model not loaded. Server is not ready."}), 503

    try:
        # Extract request data in Vertex AI format ("instances")
        data = request.json.get("instances")
        
        if not data:
            print("ERROR: No 'instances' key found in request.json.")
            return jsonify({"error": "No valid 'instances' data received"}), 400

        print(f"DEBUG: Raw input data received: {data}")

        # Convert JSON data to DataFrame
        user_df = pd.DataFrame(data)
        
        #Ensure feature names match what was used during training
        expected_features = ["Process temperature [K]", "Torque [Nm]", "Tool wear [min]", "Air temperature [K]", "Rotational speed [rpm]"]
        user_df = user_df.reindex(columns=expected_features)  # Enforces column names and order
        
        print(f"DEBUG: Converted DataFrame:\n{user_df}")

        if user_df.empty:
            print("ERROR: Input data converted to an empty DataFrame.")
            return jsonify({"error": "Input data could not be converted properly"}), 400

        # Run prediction
        failure_type = predict_failure_type(pipeline, label_encoder, user_df)

        print(f"INFO: Prediction result: {failure_type}")
        
        if not isinstance(failure_type, list):
            failure_type = [failure_type]  # Ensures Vertex AI format

        return jsonify({"predictions": failure_type}), 200

    except Exception as e:
        import traceback
        print(f"FATAL ERROR during prediction request: {e}")
        print(traceback.format_exc())  # Print full stack trace to logs
        return jsonify({"error": str(e)}), 500

@app.route("/ping", methods=["GET"])
def ping():
    # Health check should reflect whether the models are successfully loaded
    if pipeline is not None and label_encoder is not None:
        print("INFO: Health check (ping) successful - models are loaded.")
        return jsonify({"status": "healthy"}), 200
    else:
        print("ERROR: Health check (ping) failed - models are NOT loaded.")
        return jsonify({"status": "unhealthy", "reason": "models_not_loaded"}), 503

# Remember: the __main__ block should be commented out or removed for Gunicorn deployment
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))