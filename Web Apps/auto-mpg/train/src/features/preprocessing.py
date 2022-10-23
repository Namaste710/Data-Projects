<<<<<<< HEAD:Web Apps/auto-mpg/training/preprocessing.py
import pandas as pd

=======
"""
_summary_
"""
import pandas as pd

from typing import Tuple

>>>>>>> ef6b8ce6b80f8e7166e28fd3d29a73890c2634cd:Web Apps/auto-mpg/train/src/features/preprocessing.py
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Tansforms the numerical values in the Origin column to Strings
def map_origin_col(input_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        input_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    
    mapped_df = input_df.copy()
    mapped_df["Origin"] = mapped_df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return mapped_df


def num_preproc_pipeline() -> Pipeline:
    """_summary_

    Returns:
        Pipeline: _description_
    """    
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    num_pipeline = Pipeline([("imputer", imputer), 
                             ("scaler", scaler)], 
                            verbose=True)
    
    return num_pipeline

def cat_preproc_pipeline() -> Pipeline:
    """_summary_

    Returns:
        Pipeline: _description_
    """            
    ohe = OneHotEncoder()
    
    cat_pipeline = Pipeline([("one_hot_encoder", ohe)], 
                            verbose=True)
    
    return cat_pipeline

def full_preproc_ct(X_input: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """_summary_

    Args:
        X_input (pd.DataFrame): _description_

    Returns:
        tuple[pd.DataFrame, ColumnTransformer]: _description_
    """
    
    num_pipeline = num_preproc_pipeline()
    cat_pipeline = cat_preproc_pipeline()
    
    num_attributes = X_input.select_dtypes(include=["float", "int64"]).columns
    cat_attributes = X_input.select_dtypes(include=["object"]).columns

    full_pipeline = ColumnTransformer(
        [("cat", cat_pipeline, cat_attributes), 
         ("num", num_pipeline, num_attributes)],
        verbose=True,
    )
    
    preprocessed_data = full_pipeline.fit_transform(X_input)

<<<<<<< HEAD:Web Apps/auto-mpg/training/preprocessing.py
    return preprocessed_data, full_pipeline
=======
    return preprocessed_data, full_pipeline
>>>>>>> ef6b8ce6b80f8e7166e28fd3d29a73890c2634cd:Web Apps/auto-mpg/train/src/features/preprocessing.py
