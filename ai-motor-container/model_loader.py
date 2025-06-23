import joblib
import os

MODEL_BASE_DIR = os.environ.get("AIP_MODEL_DIR", os.path.join(os.getcwd(), "models"))

def load_pipeline():
    # --- CHANGE THIS LINE ---
    model_path = os.path.join(MODEL_BASE_DIR, 'failure_classifier_pipeline.joblib') # Corrected filename
    # --- END CHANGE ---

    print(f"DEBUG [model_loader]: Attempting to load pipeline from: {model_path}")
    try:
        if not os.path.exists(model_path):
            print(f"ERROR [model_loader]: Pipeline file NOT FOUND at: {model_path}")
            print(f"DEBUG [model_loader]: Contents of {MODEL_BASE_DIR}: {os.listdir(MODEL_BASE_DIR) if os.path.exists(MODEL_BASE_DIR) else 'Directory not found'}")
            return None # Indicate failure
        
        pipeline = joblib.load(model_path)
        print("INFO [model_loader]: Pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        print(f"FATAL ERROR [model_loader]: Failed to load pipeline from {model_path}. Exception: {e}")
        raise

def load_label_encoder():
    encoder_path = os.path.join(MODEL_BASE_DIR, 'label_encoder.joblib') # This one seems correct
    print(f"DEBUG [model_loader]: Attempting to load label encoder from: {encoder_path}")
    try:
        if not os.path.exists(encoder_path):
            print(f"ERROR [model_loader]: Label encoder file NOT FOUND at: {encoder_path}")
            print(f"DEBUG [model_loader]: Contents of {MODEL_BASE_DIR}: {os.listdir(MODEL_BASE_DIR) if os.path.exists(MODEL_BASE_DIR) else 'Directory not found'}")
            return None
        
        label_encoder = joblib.load(encoder_path)
        print("INFO [model_loader]: Label encoder loaded successfully.")
        return label_encoder
    except Exception as e:
        print(f"FATAL ERROR [model_loader]: Failed to load label encoder from {encoder_path}. Exception: {e}")
        raise