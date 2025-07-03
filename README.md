# Predictive AI for AC-Motor Diagnostics

**Summary:**

This solution leverages machine learning techniques to classify and predict the type of maintenance required for motors. By providing recommendations on the type of corrective maintenance needed, the system can help avoid unnecessary troubleshooting steps and misdiagnoses, ultimately reducing downtime and operational disruptions. Its development involves data collection, preprocessing, model training & optimization, deployment, validation and monitoring to build a predictive system that categorizes maintenance needs based on motor conditions and historical patterns. The outcome is a smart maintenance recommendation engine that integrates into existing workflows, enabling organizations to optimize their maintenance schedules, allocate resources efficiently, and reduce unexpected motor failures.

**Dataset:**

The data set consists of 10,000 data points stored as rows with 8 features, 1 target and 1 column with failure type, providing a robust number of samples for training and testing. The project will evaluate the predictive maintenance model using accuracy, precision, recall, and F1 score to comprehensively measure its performance in identifying equipment maintenance needs.

Dataset Link: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset 

**Project Structure:**


* Development
  * dataset  
  * notebooks → notebooks used for EDA, model development, and optimization  
  * scripts → converted notebooks and test `.py` scripts  
  * streamlit_app → front end designed for local development environment  

* Deployment  
  * ai-motor-container → docker container with inference models and supporting files  
  * ai-motor-ui-container → docker container containing front-end assets

**Pre-requisites:**
*   pandas V 2.2.3
*   scikit-learn V 1.6.1
*   joblib V 1.5.0
*   gunicorn V 23.0.0
*   imbalanced-learn V 0.13.0
*   flask V 3.1.0
*   streamlit V 1.45.1
*   requests V 2.32.3
*   google-auth V 2.40.3

**Model Details:**
XGBoost model predicting type of errors based on sensor data related to AC-Motors.

**App Preview:**

![Screenshot 2025-06-22 215558](https://github.com/user-attachments/assets/513a136c-c336-4638-8f3e-881960f682e6)


**Deployment Architecture:**
![image](https://github.com/user-attachments/assets/e3a991ff-cb47-4e0c-8074-0e9750449759)

**Request-Response Logging in Google BigQuery:**
![image](https://github.com/user-attachments/assets/bc7caf4d-862c-4356-b657-a5fdc83887f8)


**Front-End Application URL:** https://ai-motor-ui-container-1068549159816.us-central1.run.app/
