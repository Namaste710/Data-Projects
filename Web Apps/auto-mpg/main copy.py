"""
_summary_
"""

from fastapi import FastAPI
import uvicorn

from typing import Dict

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: float
    
app = FastAPI(debug=True, description="Serve MPG prediction model")

@app.get("/health", status_code=200)
def health_check() -> Dict[str, str]:
    """
    Actuator like health endpoint.
    Returns:
    (dict of str: str): Health status object.
    """
    return {"Status": "Healthy"}

@app.post("/predict", status_code=200)
def predict(request: PredictionRequest):
    print("hello")
    # map

    # if type(input_data) == dict:
    #     df = pd.DataFrame(input_data)
    # else:
    #     df = input_data

    # df = map_origin_col(df)
    # df = full_preproc_ct(df)
    # y_pred = model.predict(df)
    # return y_pred

if __name__ == '__main__':
    
   uvicorn.run(app, host="0.0.0.0", port=80)


