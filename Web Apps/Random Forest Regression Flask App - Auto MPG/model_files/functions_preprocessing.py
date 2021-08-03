import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Tansforms the numerical values in the Origin column to Strings
def preprocess_origin_col(df):
    df['Origin'] = df['Origin'].map({1: 'India', 2: 'USA', 3: 'Germany'})
    return df
	
# Creates two new features out of the existing ones
acceleration_pos, horsepower_pos, cylinders_pos = 4, 2, 0
class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): 
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        acc_on_cyl = X[:, acceleration_pos] / X[:, horsepower_pos]
        if self.acc_on_power:
            acc_on_power = X[:, acceleration_pos] / X[:, horsepower_pos]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]
		
def numerical_pipeline_transformer(df):
    '''Preprocesses numerical columns in the DataFrame

    Args:
        df: DataFrame
    
    Returns:
        numerical_attr: DataFrame with only numerical columns
        numerical_pipeline: The pipeline object
    '''
    numerical = ['float', 'int64']

    numerical_data = df.select_dtypes(include=numerical)

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_creator', FeatureCreator()),
        ('std_scaler', StandardScaler())
    ])
    return numerical_data, numerical_pipeline

def full_pipeline(df):
    '''Completely preprocesses the DataFrame (numerical and categorical columns)
    
    Args:
        df: DataFrame

    Returns:
        preprocessed_data: Preprocessed DataFrame
    '''
    numerical_attributes, numerical_pipeline = numerical_pipeline_transformer(df)
    numerical_attributes = list(numerical_attributes)
    cat_attributes = ['Origin']

    full_pipeline = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_attributes),
        ('cat', OneHotEncoder(), cat_attributes)
    ])
    preprocessed_data = full_pipeline.fit_transform(df)
    return preprocessed_data
    
def predict_y(input_data, model):
    '''Predicts values for the input data and model
    
    Args:
        input_data: DataFrame or dictionary
        model: Pickled ML-Model
    Returns:
        y_pred: List of predicted values for the input data
    '''
    if type(input_data) == dict:
        df = pd.DataFrame(input_data)
    else:
        df = input_data
    
    df = preprocess_origin_col(df)
    df = full_pipeline(df)
    y_pred = model.predict(df)
    return y_pred