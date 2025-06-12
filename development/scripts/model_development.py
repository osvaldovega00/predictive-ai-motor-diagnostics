import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import joblib
import h2o
from h2o.automl import H2OAutoML
import tensorflow as tf
from tensorflow import keras
from keras.api.layers import Input, Dense, Dropout
from keras.api.models import Model
from keras.api.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

h2o.init()

# Load dataset
df = pd.read_csv("00-AI4I 2020 Predictive Maintenance Dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=["UDI", "Product ID"])

# Define categorical & numerical columns
categorical_columns = ["Type"]
numerical_columns = ["Process temperature [K]", "Torque [Nm]", "Tool wear [min]", "Air temperature [K]", "Rotational speed [rpm]"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numerical_columns), ("cat", OneHotEncoder(drop="first"), categorical_columns)])

# Drop 'Failure Type' to prevent data leakage
X_occurrence = df.drop(columns=["Target", "Failure Type"])
y_occurrence = df["Target"]

# Split data
X_train_occ, X_test_occ, y_train_occ, y_test_occ = train_test_split(X_occurrence, y_occurrence, test_size=0.25, random_state=42)

occurrence_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(class_weight="balanced", random_state=42))])

# Train and evaluate
occurrence_pipeline.fit(X_train_occ, y_train_occ)

y_pred_occ = occurrence_pipeline.predict(X_test_occ)
accuracy = accuracy_score(y_test_occ, y_pred_occ)
cm = confusion_matrix(y_test_occ, y_pred_occ)

print(f"Accuracy: {accuracy:.4f}\n")
print("Occurrence Model Performance:\n", classification_report(y_test_occ, y_pred_occ))
print("ROC AUC Score:", roc_auc_score(y_test_occ, occurrence_pipeline.predict_proba(X_test_occ)[:, 1]), "\n")
print("Confusion Matrix:\n", cm)

# Extract failure cases
failure_subset = df[df["Target"] == 1].copy()

# Drop 'Target' to prevent leakage
X_failure = failure_subset.drop(columns=["Target", "Failure Type"])
y_failure = failure_subset["Failure Type"]

# Encode failure type labels
label_encoder = LabelEncoder()
y_failure_encoded = label_encoder.fit_transform(y_failure)

# Split data
X_train_fail, X_test_fail, y_train_fail, y_test_fail = train_test_split(X_failure, y_failure_encoded, test_size=0.25, random_state=42)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

failure_type_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))])

# Train and evaluate
failure_type_pipeline.fit(X_train_fail, y_train_fail)

y_pred_fail = failure_type_pipeline.predict(X_test_fail)
accuracy_fail = accuracy_score(y_test_fail, y_pred_fail)

print(f"Accuracy: {accuracy_fail:.4f}\n")
print("Failure Type Model Performance:\n", classification_report(y_test_fail, y_pred_fail, target_names=label_encoder.classes_))

df = h2o.import_file("00-AI4I 2020 Predictive Maintenance Dataset.csv")

# Drop irrelevant columns
df = df.drop(["\ufeffUDI", "Product ID"], axis=1)

# Convert categorical variables
df["Type"] = df["Type"].asfactor()
df["Failure Type"] = df["Failure Type"].asfactor()
df["Target"] = df["Target"].asfactor()

# Split dataset
train, test = df.split_frame(ratios=[0.75], seed=42)

# Drop 'Failure Type' (to prevent leakage)
X_occurrence = train.drop(["Target", "Failure Type"])
y_occurrence = train["Target"]

# Run H2O AutoML
aml_occ = H2OAutoML(max_models=20, seed=42)
aml_occ.train(x=X_occurrence.columns, y="Target", training_frame=train)

# View leaderboard
print(aml_occ.leaderboard)

# Get the best model from AutoML
best_model = aml_occ.leader

# Print model performance
print(best_model.model_performance().show())

# Generate predictions on the test set
pred_occurrence = aml_occ.leader.predict(test)

# Convert predictions to a Pandas dataframe for easier analysis
pred_occurrence_df = pred_occurrence.as_data_frame()
test_df = test.as_data_frame()

# Merge predictions with actual values
results_occ = test_df[["Target"]].copy()
results_occ["Predicted_Target"] = pred_occurrence_df["predict"]
print(results_occ.head(10))

# Keep only failure cases
failure_subset = train[train["Target"] == "1"]

# Drop 'Target' (to prevent leakage)
X_failure = failure_subset.drop(["Target", "Failure Type"])
y_failure = failure_subset["Failure Type"]

# Run H2O AutoML
aml_fail = H2OAutoML(max_models=20, seed=42)
aml_fail.train(x=X_failure.columns, y="Failure Type", training_frame=failure_subset)

# View leaderboard
print(aml_fail.leaderboard)

# Get the best model from AutoML
best_model = aml_fail.leader

# Print detailed model performance
print(best_model.model_performance().show())

# Filter test data where failure occurred (Target = 1)
failure_test_subset = test[test["Target"] == "1"]

# Generate predictions for failure type classification
pred_failure_type = aml_fail.leader.predict(failure_test_subset)

# Convert predictions to a Pandas dataframe
pred_failure_df = pred_failure_type.as_data_frame()
failure_test_df = failure_test_subset.as_data_frame()

# Merge predictions with actual failure types
results_failure = failure_test_df[["Failure Type"]].copy()
results_failure["Predicted_Failure_Type"] = pred_failure_df["predict"]
print(results_failure.head(10))

# Load dataset
df = pd.read_csv("00-AI4I 2020 Predictive Maintenance Dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=["UDI", "Product ID"])

# Encode categorical features
df["Type"] = df["Type"].astype("category").cat.codes

# Encode target variables
label_encoder = LabelEncoder()
df["Failure Type"] = label_encoder.fit_transform(df["Failure Type"])
df["Target"] = df["Target"].astype("int")

# Separate features for both models
X = df.drop(columns=["Target", "Failure Type"])
y_occurrence = df["Target"]
y_failure = df[df["Target"] == 1]["Failure Type"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode failure type labels
y_failure_encoded = to_categorical(y_failure)

# Split data
X_train_occ, X_test_occ, y_train_occ, y_test_occ = train_test_split(X_scaled, y_occurrence, test_size=0.25, random_state=42)
X_train_fail, X_test_fail, y_train_fail, y_test_fail = train_test_split(X_scaled[y_occurrence == 1], y_failure_encoded, test_size=0.25, random_state=42)

# Define input shape
input_shape = X_train_occ.shape[1]

# Shared Input Layer
inputs = Input(shape=(input_shape,))
x = Dense(128, activation="relu")(inputs)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)

# Output 1: Binary Classification (Failure Occurrence)
output_occ = Dense(1, activation="sigmoid", name="occurrence_output")(x)

# Output 2: Multi-Class Classification (Failure Type)
output_fail = Dense(y_failure_encoded.shape[1], activation="softmax", name="failure_type_output")(x)

# Define Model
model = Model(inputs=inputs, outputs=[output_occ, output_fail])

# Compile Model
model.compile(optimizer="adam",
              loss={"occurrence_output": "binary_crossentropy", "failure_type_output": "categorical_crossentropy"},
              metrics={"occurrence_output": "accuracy", "failure_type_output": "accuracy"})

# Print Model Summary
print(model.summary())

# Train using both tasks
X_train_occ_filtered = X_train_occ[y_train_occ == 1]

# Ensure X_train_occ_filtered and y_train_fail have the same number of samples
X_train_occ_filtered = X_train_occ_filtered[:len(y_train_fail)]

# Now train the model
history = model.fit(X_train_occ_filtered,  # Use filtered X_train_occ
                    {"occurrence_output": y_train_occ[y_train_occ == 1][:len(y_train_fail)],
                     "failure_type_output": y_train_fail},
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

# Generate predictions for failure occurrence
pred_occurrence = model.predict(X_test_occ)

# Extract binary predictions
pred_occ_binary = (pred_occurrence[0] > 0.5).astype(int)

# Compare predictions with actual values
results_occ = pd.DataFrame({"Actual_Target": y_test_occ, "Predicted_Target": pred_occ_binary.flatten()})

# Print sample predictions
print(results_occ.head())

# Generate predictions for failure type classification (only for Target = 1)
pred_failure_type = model.predict(X_test_fail)

# Convert softmax outputs to class labels
pred_fail_labels = np.argmax(pred_failure_type[1], axis=1)

# Compare predictions with actual failure types
results_fail = pd.DataFrame({
    "Actual_Failure_Type": np.argmax(y_test_fail, axis=1),
    "Predicted_Failure_Type": pred_fail_labels
})

# Decode failure type labels back to original categories
results_fail["Actual_Failure_Type"] = label_encoder.inverse_transform(results_fail["Actual_Failure_Type"])
results_fail["Predicted_Failure_Type"] = label_encoder.inverse_transform(results_fail["Predicted_Failure_Type"])

# Print sample predictions
print(results_fail.head())

# Accuracy of failure occurrence prediction
acc_occ = accuracy_score(y_test_occ, pred_occ_binary)
print(f"Accuracy: {acc_occ:.4f}\n")

# Classification report for failure type prediction
print("Performance:\n", classification_report(results_fail["Actual_Failure_Type"], results_fail["Predicted_Failure_Type"]))
