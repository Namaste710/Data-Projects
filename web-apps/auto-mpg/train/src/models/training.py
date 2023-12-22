"""
_summary_
"""
import numpy as np
import pandas as pd

import features.preprocessing as pp

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import logging

import joblib

from constants import FILE_PATH, MODEL_PATH

from data.make_data import load_data

def model_training(input_df: pd.DataFrame):
    """_summary_

    Args:
        input_df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """    
    input_df = pp.map_origin_col(input_df)
    
    X = input_df.drop("MPG", axis=1)
    y = input_df["MPG"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    X_train, full_pl = pp.full_preproc_ct(X_train)
    
    X_test = full_pl.transform(X_test)
    
    model = LinearRegression().fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    
    logging.info(f"\tMSE: {mean_squared_error(y_predict, y_test)}")
    logging.info(f"\tRMSE: {np.sqrt(mean_squared_error(y_predict, y_test))}")
    logging.info(f"\tR²: {r2_score(y_predict, y_test)}")

    return model
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    df = load_data(FILE_PATH)
    
    model = model_training(df)
    
    joblib.dump(model, MODEL_PATH)

