import requests
import pandas as pd
import json

# === Define Local API Endpoint (Docker Container) ===
url = "http://127.0.0.1:8080/predict"  # Targeting your locally running container

# === Prepare Input Data (Structured Like Vertex AI) ===
input_values = {
    "Process temperature [K]": 0,
    "Torque [Nm]": 0,
    "Tool wear [min]": 0,
    "Air temperature [K]": 0,
    "Rotational speed [rpm]": 0,
}

payload = {"instances": [input_values]}  # Keeping the structure consistent with Vertex AI

# === Debug: Print JSON Structure Before Sending ===
print("Formatted JSON:", json.dumps(payload, indent=2))

# === Send Request to Local Flask API ===
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

    # Print Response Details
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Raw Response:", response.text)

    # Extract Predictions
    response_json = response.json()
    if "predictions" in response_json:
        print("Prediction Output:", response_json["predictions"])
    else:
        print("⚠️ 'predictions' key not found in response.")

except requests.exceptions.ConnectionError as e:
    print(f"ERROR: Could not connect to {url}. Make sure your Docker container is running.")
    print(f"Details: {e}")
except requests.exceptions.Timeout as e:
    print("ERROR: Request timed out. This could mean the container is slow to respond.")
    print(f"Details: {e}")
except requests.exceptions.RequestException as e:
    print(f"ERROR: An error occurred: {e}")