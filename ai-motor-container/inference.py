import pandas as pd
import joblib

def predict_failure_type(pipeline, label_encoder, input_df):
    """
    Predict failure type using the classifier pipeline and return decoded label.
    If 'No Failure' is predicted, returns that; otherwise returns the failure type.
    """
    prediction = pipeline.predict(input_df)
    decoded = label_encoder.inverse_transform(prediction)
    return decoded[0]