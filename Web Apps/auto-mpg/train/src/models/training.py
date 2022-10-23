"""
_summary_
"""
import numpy as np
import pandas as pd
<<<<<<< HEAD:Web Apps/auto-mpg/training/training.py
import preprocessing as pp
=======

import features.preprocessing as pp

>>>>>>> ef6b8ce6b80f8e7166e28fd3d29a73890c2634cd:Web Apps/auto-mpg/train/src/models/training.py
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
<<<<<<< HEAD:Web Apps/auto-mpg/training/training.py

import logging
logging.getLogger().setLevel(logging.INFO)

def load_data(input_path: str) -> pd.DataFrame:
    """_summary_

    Args:
        input_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    cols = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'ModelYear', 'Origin']
=======

import logging
>>>>>>> ef6b8ce6b80f8e7166e28fd3d29a73890c2634cd:Web Apps/auto-mpg/train/src/models/training.py

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

<<<<<<< HEAD:Web Apps/auto-mpg/training/training.py
    return model, full_pl


=======
    return model
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    df = load_data(FILE_PATH)
    
    model = model_training(df)
    
    joblib.dump(model, MODEL_PATH)
>>>>>>> ef6b8ce6b80f8e7166e28fd3d29a73890c2634cd:Web Apps/auto-mpg/train/src/models/training.py

