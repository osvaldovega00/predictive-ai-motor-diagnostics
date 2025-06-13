import requests
import pandas as pd

# Define the API endpoint
url = "http://127.0.0.1:8080/predict"

# Define input data
input_values = {
    "Torque [Nm]": 0,
    "Air temperature [K]": 0,
    "Process temperature [K]": 0,
    "Rotational speed [rpm]": 0,
    "Tool wear [min]": 0
}

# Convert input data into a DataFrame
user_df = pd.DataFrame([input_values])

# Convert DataFrame to JSON format (for API request)
json_data = user_df.to_dict(orient="records")

# Send request to Flask API
response = requests.post(url, json={"input_data": json_data})

# Print response details
print("Status Code:", response.status_code)
print("Raw Response:", response.text)