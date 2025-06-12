import pandas as pd
import h2o

def predict_occ_model(occ_model, user_h2o):
    """Occurence prediction using H2O"""
    pred_occ = occ_model.predict(user_h2o)
    return pred_occ.as_data_frame()["predict"][0]

def predict_fail_model(fail_model, user_h2o):
    """Occurence prediction using H2O"""
    pred_fail = fail_model.predict(user_h2o)
    df_fail = pred_fail.as_data_frame()
    df_fail_sorted = df_fail.iloc[:, 1:].T.sort_values(by=0, ascending=False)
    failure_type_outcome_1 = df_fail_sorted.index[0]
    #failure_type_outcome_2 = df_fail_sorted.index[1]
    return failure_type_outcome_1
 
    