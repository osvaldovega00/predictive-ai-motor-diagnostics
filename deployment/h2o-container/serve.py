from flask import Flask, request, jsonify
import h2o
import pandas as pd
from model_loader import load_occ_model, load_fail_model
from inference import predict_occ_model, predict_fail_model

# Initialize Flask
app = Flask(__name__)

#Start H2O once and prevent auto-closing
h2o.init()
h2o.no_progress()

# Load models **only once** for efficiency
occ_model = load_occ_model()
fail_model = load_fail_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("input_data")  # Extract JSON input
        if not data:
            return jsonify({"error": "No valid input data received"}), 400
        
        print("Received input data:", data)  # Debugging print

        # Convert JSON to DataFrame
        user_df = pd.DataFrame(data)
        print("Converted to DataFrame:", user_df)  # Debugging print

        # Ensure DataFrame isn't empty
        if user_df.empty:
            return jsonify({"error": "Input data could not be converted properly"}), 400
        
        # Convert to H2OFrame
        user_h2o = h2o.H2OFrame(user_df)
        print("Converted to H2OFrame")  # Debugging print

        # Run predictions
        occ_result = predict_occ_model(occ_model, user_h2o)
        fail_result = predict_fail_model(fail_model, user_h2o)

        return jsonify({
            "occurrence_prediction": int(occ_result),  # Convert int64 to int
            "failure_type_prediction": str(fail_result)  # Ensure it's a JSON-serializable string
            })  

    except Exception as e:
        print("ERROR:", str(e))  # Print actual error
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)