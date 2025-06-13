import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AI for AC Motors", layout="wide")

FLASK_API_URL = "http://127.0.0.1:8080/predict"

if "motorSize" not in st.session_state:
    st.session_state["motorSize"] = "Small: 1HP or less"
if "toolWear" not in st.session_state:
    st.session_state["toolWear"] = 0
if "torque" not in st.session_state:
    st.session_state["torque"] = 0
if "rotSpeed" not in st.session_state:
    st.session_state["rotSpeed"] = 0
if "proTemp" not in st.session_state:
    st.session_state["proTemp"] = 0
if "airTemp" not in st.session_state:
    st.session_state["airTemp"] = 0
if "status_message" not in st.session_state:
    st.session_state["status_message"] = "Prediction:"

with st.sidebar:
    st.markdown("### App Status")
    st.success("✅ AI-powered prediction ready")
    st.info("ℹ️ Models loaded successfully")
    
    st.markdown("---")
    
    st.markdown("### User Inputs")
    st.write(f"Motor Size: **{st.session_state['motorSize']}**")
    st.write(f"Tool Wear: **{st.session_state['toolWear']} min**")
    st.write(f"Torque: **{st.session_state['torque']} Nm**")
    st.write(f"Rotational Speed: **{st.session_state['rotSpeed']} rpm**")
    st.write(f"Process Temperature: **{st.session_state['proTemp']} K**")
    st.write(f"Room Temperature: **{st.session_state['airTemp']} K**")

    st.markdown("---")

    st.markdown("### Version Info")
    st.write("**Version:** 1.0.0")
    st.write("**Updated:** June 2025")

# Function to update status message
def update_status(new_message):
    st.session_state["status_message"] = new_message

# Function to reset inputs and update status message
def reset_inputs():
    st.session_state["motorSize"] = "Small: 1HP or less"
    st.session_state["toolWear"] = 0
    st.session_state["torque"] = 0
    st.session_state["rotSpeed"] = 0
    st.session_state["proTemp"] = 0
    st.session_state["airTemp"] = 0
    update_status("Prediction:")

# Function for submit button logic
def submit_inputs():
    # Prepare input data
    input_values = {
        "Torque [Nm]": st.session_state["torque"],
        "Air temperature [K]": st.session_state["airTemp"],
        "Process temperature [K]": st.session_state["proTemp"],
        "Rotational speed [rpm]": st.session_state["rotSpeed"],
        "Tool wear [min]": st.session_state["toolWear"]
    }
    
    # Send request to Flask container
    response = requests.post(FLASK_API_URL, json={"input_data": [input_values]})

    if response.status_code == 200:
        prediction = response.json()
        update_status(f"Prediction: {prediction['failure_type_prediction'] if prediction['occurrence_prediction'] else 'Motor data does not show signs of failure.'}")
    else:
        update_status("Prediction: Error in retrieving predictions.")

#Features column, static list
feature_names = ["Torque [Nm]", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Tool wear [min]"]

#Global styiling
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

    # Input fields
    st.selectbox("Select Motor Size:", ["Small: 1HP or less", "Medium: 1HP-500HP", "Large: 500HP or more"], index=0, key="motorSize")
    st.number_input("Enter Tool Wear [min]", min_value=0, max_value=None, step=1, key="toolWear")
    st.number_input("Enter Torque [Nm]", min_value=0, max_value=300, step=1, key="torque")
    st.number_input("Enter Rotational Speed [rpm]", min_value=0, max_value=4000, step=10, key="rotSpeed")
    st.number_input("Enter Process Temperature [K]", min_value=0, max_value=500, step=10, key="proTemp")
    st.number_input("Enter Room Temperature [K]", min_value=0, max_value=500, step=10, key="airTemp")

    # Single column for buttons
    button_col = st.columns([1, 1])
    with button_col[0]:
        st.button("Submit", use_container_width=True, on_click=submit_inputs)
    with button_col[1]:
        st.button("Cancel", use_container_width=True, on_click=reset_inputs)

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
    <h2 style="text-align: center;">GitHub Repository</h2>
    <p style="text-align: center;">
    Explore the full project and source codes on GitHub:<br>
    <a href="https://github.com/osvaldovega00/Predictive-AI-for-AC-Motor-Diagnostics" target="_blank">
    <strong>Predictive AI for AC Motor Diagnostics</strong></a>
    </p>
    </div>
    """, unsafe_allow_html=True)
