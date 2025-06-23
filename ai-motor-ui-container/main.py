import streamlit as st
import requests
import json
from google.auth import default
from google.auth.transport.requests import Request

# === set_page_config ===
st.set_page_config(page_title="AI for AC Motors", layout="wide")

# === CONFIGURATION ===
PROJECT_ID = "ai-motor-diagnostics"
ENDPOINT_ID = "2596796875667406848"
REGION = "us-central1"

# Define a function to get access token dynamically
@st.cache_data(ttl=3600)  # Cache the token for 1 hour
def get_access_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token
access_token = get_access_token()

# === Step 2: Define the Vertex AI endpoint URL ===
VERTEX_AI_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"

# === Session State Defaults ===
if "status_message" not in st.session_state:
    st.session_state["status_message"] = "Prediction:"

with st.sidebar:
    st.markdown("### App Status")
    st.success("✅ AI-powered prediction ready")    
    st.markdown("---")    
    st.markdown("### Current Inputs")
    st.write(f"Motor Size: **{st.session_state.get('motorSize_form', 'N/A')}**")
    st.write(f"Tool Wear: **{st.session_state.get('toolWear_form', 0)} min**")
    st.write(f"Torque: **{st.session_state.get('torque_form', 0)} Nm**")
    st.write(f"Rotational Speed: **{st.session_state.get('rotSpeed_form', 0)} rpm**")
    st.write(f"Process Temperature: **{st.session_state.get('proTemp_form', 0)} K**")
    st.write(f"Room Temperature: **{st.session_state.get('airTemp_form', 0)} K**")
    st.markdown("---")
    st.markdown("### Version Info")
    st.write("**Version:** 1.0.0")
    st.write("**Updated:** June 2025")

# Function to update status message (still useful)
def update_status(new_message):
    st.session_state["status_message"] = new_message

# Function to reset inputs and update status message
def reset_inputs():
    update_status("Prediction:")

#Global styling
st.markdown("""<style>div[data-testid="stTabs"]{display: flex;justify-content: center;}</style>""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Home", "About"])

with tab1:
    st.markdown("<h1 style='text-align: center;'>Predictive AI for AC-Motor Diagnostics</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="width: 800px; margin: auto; text-align: justify;">
    Select the motor size and enter the last recorded parameters of the motor. Once all inputs are provided, click 
    Submit to obtain a prediction on the possible type of fault. If you wish to reset the inputs, click Cancel.
    </div>
    """, unsafe_allow_html=True)

    # Placeholder for status message
    status_placeholder = st.empty()

    # Display default or dynamic message with custom styling
    status_placeholder.markdown(
    f"""
    <div style='text-align: center; font-size: 20px; font-weight: bold;'>
        {st.session_state.get("status_message", "Prediction:")}
    </div>
    """, unsafe_allow_html=True)

    # --- Start of the Form ---
    with st.form(key="prediction_form"):
        # Input fields
        # Give them distinct keys for the form to manage their state
        st.session_state["motorSize_form"] = st.selectbox("Select Motor Size:", ["Small: 1HP or less", "Medium: 1HP-500HP", "Large: 500HP or more"], index=0, key="motorSize_input")
        st.session_state["toolWear_form"] = st.number_input("Enter Tool Wear [min]", min_value=0, max_value=None, step=1, key="toolWear_input")
        st.session_state["torque_form"] = st.number_input("Enter Torque [Nm]", min_value=0, max_value=300, step=1, key="torque_input")
        st.session_state["rotSpeed_form"] = st.number_input("Enter Rotational Speed [rpm]", min_value=0, max_value=4000, step=10, key="rotSpeed_input")
        st.session_state["proTemp_form"] = st.number_input("Enter Process Temperature [K]", min_value=0, max_value=500, step=10, key="proTemp_input")
        st.session_state["airTemp_form"] = st.number_input("Enter Room Temperature [K]", min_value=0, max_value=500, step=10, key="airTemp_input")

        # Single column for buttons
        button_col = st.columns([1, 1])
        with button_col[0]:
            submit_button = st.form_submit_button("Submit", use_container_width=True)
        with button_col[1]:
            cancel_button = st.form_submit_button("Cancel", use_container_width=True)

        # Logic for buttons
        if submit_button:
            if access_token:
                input_values = {
                    "Process temperature [K]": st.session_state["proTemp_form"],
                    "Torque [Nm]": st.session_state["torque_form"],
                    "Tool wear [min]": st.session_state["toolWear_form"],
                    "Air temperature [K]": st.session_state["airTemp_form"],
                    "Rotational speed [rpm]": st.session_state["rotSpeed_form"]
                }
                
                # === Step 4: Send request ===
                headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
                try:
                    response = requests.post(VERTEX_AI_URL, headers=headers, json={"instances": [input_values]})

                    if response.status_code == 200:
                        response_json = response.json()
                        predictions = response_json.get("predictions", [])

                        if predictions:
                            update_status(f"Prediction: {predictions[0]}")
                        else:
                            update_status("Prediction: No predictions found.")
                    else:
                        update_status(f"Prediction: Error retrieving predictions. Status code: {response.status_code}, Response: {response.text}")
                except requests.exceptions.RequestException as e:
                    update_status(f"Prediction: Network or API error: {e}")
            else:
                update_status("Prediction: Access token not available. Cannot make prediction.")
            
            status_placeholder.markdown(
                f"""
                <div style='text-align: center; font-size: 20px; font-weight: bold;'>
                    {st.session_state["status_message"]}
                </div>
                """, unsafe_allow_html=True
            )
        
        if cancel_button:
            update_status("Prediction:")
            status_placeholder.markdown(
                f"""
                <div style='text-align: center; font-size: 20px; font-weight: bold;'>
                    {st.session_state["status_message"]}
                </div>
                """, unsafe_allow_html=True
            )


with tab2:
    st.markdown("<h1 style='text-align: center;'>Predictive AI for AC-Motor Diagnostics</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="width: 800px; margin: auto; text-align: justify;">
    Maintaining AC motors in optimal working condition is essential for ensuring the smooth operation of industrial processes. 
    Unexpected motor failures can lead to significant downtime, increased operational costs, and safety hazards. By leveraging 
    AI and machine learning techniques, this system classifies motor conditions and predicts the type of maintenance required. 
    By providing targeted corrective maintenance recommendations, it helps avoid unnecessary troubleshooting and misdiagnoses, 
    ultimately reducing downtime and operational disruptions. This model analyzes motor performance data from various sensors to identify 
    patterns and indicators, offering recommendations on the type of corrective service needed. This solution aims to demonstrate the 
    potential of AI in industrial settings. While its predictions can assist in maintenance planning, they should always be verified and 
    assessed before implementation. The development team is not responsible for any decisions made based on these predictions. Proper 
    evaluation by qualified personnel is essential to ensure safe and effective outcomes.   
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="width: 800px; margin: auto; text-align: justify;">
    <h2 style="text-align: center;'>GitHub Repository</h2>
    <p style="text-align: center;'>
    Explore the full project and source codes on GitHub:<br>
    <a href="https://github.com/osvaldovega00/Predictive-AI-for-AC-Motor-Diagnostics" target="_blank">
    <strong>Predictive AI for AC Motor Diagnostics</strong></a>
    </p>
    </div>
    """, unsafe_allow_html=True)