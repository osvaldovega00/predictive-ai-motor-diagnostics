import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

df = h2o.import_file("AI4I 2020 Predictive Maintenance Dataset.csv")

# Drop irrelevant columns
df = df.drop(["\ufeffUDI", "Product ID", "Type"], axis=1)

# Convert categorical variables
#df["Type"] = df["Type"].asfactor()
df["Failure Type"] = df["Failure Type"].asfactor()
df["Target"] = df["Target"].asfactor()

# Split dataset
train, test = df.split_frame(ratios=[0.75], seed=42)

# Drop 'Failure Type' (to prevent leakage)
X_occurrence = train.drop(["Target", "Failure Type"])
y_occurrence = train["Target"]

# Run H2O AutoML
aml_occ = H2OAutoML(max_models=20, balance_classes=True, stopping_metric="AUC", seed=42)
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

print(train["Target"].table())
print(failure_subset["Failure Type"].table())

metalearner = best_model.metalearner() # Get the metalearner object using metalearner() method
metalearner.varimp_plot() # Call varimp_plot on the metalearner object

# Instead of calling best_model.base_models(), directly access the list of base models
base_models = best_model.base_models  # Access the base_models attribute

# Access a specific base model (e.g., the first one):
# Get the model ID of the first base model
gbm_model_id = base_models[0]

# Get the actual model object using h2o.get_model()
gbm_model = h2o.get_model(gbm_model_id)

importance_df = gbm_model.varimp(use_pandas=True)  # Converts to Pandas DataFrame
print(importance_df)

# Get variable importance from the base model:
gbm_model.varimp_plot()

# Ensure models are correctly assigned
model_occ = aml_occ.leader
model_fail = aml_fail.leader

# Save MOJO models
model_path_occ = model_occ.save_mojo(path="failure_occurrence_model.zip", force=True)
model_path_fail = model_fail.save_mojo(path="failure_type_model.zip", force=True)

# Define Drive paths
model_drive_path_occ = "/content/drive/MyDrive/failure_occurrence_model.zip"
model_drive_path_fail = "/content/drive/MyDrive/failure_type_model.zip"

# Save models in Drive
model_occ.save_mojo(path=model_drive_path_occ, force=True)
model_fail.save_mojo(path=model_drive_path_fail, force=True)

# Print confirmation
print(f"Occurrence model saved at: {model_drive_path_occ}")
print(f"Failure type model saved at: {model_drive_path_fail}")

# Define the paths based on the uploaded files
model_occ_path = "/content/drive/MyDrive/failure_occurrence_model.zip"
model_fail_path = "/content/drive/MyDrive/failure_type_model.zip"

# Load models from local Colab storage
model_occ = h2o.import_mojo(model_occ_path)
model_fail = h2o.import_mojo(model_fail_path)

# Initialize empty input list
input_values = []

# Sequential user input prompts
print("Enter current motor parameters\n")
input_values.append(float(input("Torque [Nm]: ")))
input_values.append(float(input("Air Temperature [K]: ")))
input_values.append(float(input("Process Temperature [K]: ")))
input_values.append(float(input("Rotational Speed [rpm]: ")))
input_values.append(float(input("Tool Wear [min]: ")))

# Define feature names (including dropdown selection)
feature_names = ["Torque [Nm]", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Tool wear [min]"]

# Create a Pandas DataFrame with user input
user_df = pd.DataFrame([input_values], columns=feature_names)
# Convert to H2OFrame for model predictions
user_h2o = h2o.H2OFrame(user_df)
print(user_df.head())

pred_occ = model_occ.predict(user_h2o)
failure_occurrence = pred_occ.as_data_frame()["predict"][0]

if failure_occurrence == 0:
    occurence_outcome = "No failure predicted"
else:
    occurence_outcome = "Failure detected"

if failure_occurrence == 1:
    pred_fail_type = model_fail.predict(user_h2o)
    df_fail = pred_fail_type.as_data_frame()
    df_fail_sorted = df_fail.iloc[:, 1:].T.sort_values(by=0, ascending=False)
    failure_type_outcome_1 = df_fail_sorted.index[0]
    failure_type_outcome_2 = df_fail_sorted.index[1]

if failure_occurrence == 1:
  print("Motor shows signs of failure, possible causes are: " + failure_type_outcome_1 + ' or ' + failure_type_outcome_2)
else:
  print(occurence_outcome)