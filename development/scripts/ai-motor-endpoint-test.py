import requests
import pandas as pd
import subprocess
import json

# === CONFIGURATION ===
project_id = "ai-motor-diagnostics"
endpoint_id = "2596796875667406848"
region = "us-central1"

# === Step 1: Get access token ===
access_token = subprocess.check_output([
    r"C:\Users\osval\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
    "auth",
    "print-access-token"
]).decode("utf-8").strip()

# === Step 2: Define the endpoint URL ===
url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"

# === Step 3: Prepare input data ===
input_values = {
    "Process temperature [K]": 0,
    "Torque [Nm]": 0,
    "Tool wear [min]": 0,
    "Air temperature [K]": 0,
    "Rotational speed [rpm]": 0,
}

# Ensure single-instance format for Vertex AI
payload = {"instances": [input_values]}  # Wrap input in a list

# === Step 4: Send request ===
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

print("Payload being sent:", json.dumps(payload, indent=2))  # Debugging

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    response_json = response.json()
    predictions = response_json.get("predictions", [])

    if predictions:
        print("Prediction Output:", predictions[0])  # Extract single output
    else:
        print("No predictions found in response:", response_json)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)