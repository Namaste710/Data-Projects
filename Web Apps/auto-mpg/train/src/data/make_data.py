import pandas as pd

from constants import COLUMN_NAMES

def load_data(input_path: str) -> pd.DataFrame:
    
    # reading the .data file 
    df = pd.read_csv(input_path, na_values='?', names=COLUMN_NAMES, comment='\t', sep=' ', skipinitialspace=True)

    return df