import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.typing import Literal
from training import preprocessing as pp

MODEL_PATH = "artifacts/model.joblib"


class Car(BaseModel):
    # capital letters because needed as column names in preprocessing
    Cylinders: int = None
    Displacement: float = None
    Horsepower: float = None
    Weight: float = None
    Acceleration: float = None
    ModelYear: int = None
    Origin: Literal[1, 2, 3]


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_y(input_data: Car):
    """_summary_

    Args:
        input_data (Car): _description_

    Returns:
        _type_: _description_
    """

    transposed_input_data = pd.DataFrame([input_data.dict()])

    model, preproc_pl = joblib.load(MODEL_PATH)

    df_mapped = pp.map_origin_col(transposed_input_data)

    df_full = preproc_pl.transform(df_mapped)

    y_pred = model.predict(df_full).tolist()

    return y_pred
