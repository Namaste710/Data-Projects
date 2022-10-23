"""
_summary_
"""
from flask import Flask, request, jsonify

from fastapi import FastAPI

import pickle
from model_files.preprocessing import predict_y
from model_files.training import model_training, load_data


app = FastAPI("MPG predictor")
# Initializing Flask app
app = Flask('app')

# Testroute
@app.route('/test', methods=['GET'])
def test():
    return 'Ping!'


def predict_y(input_data, model):

    if type(input_data) == dict:
        df = pd.DataFrame(input_data)
    else:
        df = input_data

    df = map_origin_col(df)
    df = full_preproc_ct(df)
    y_pred = model.predict(df)
    return y_pred

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Getting the input vehicle data as a JSON ds
    vehicle = request.get_json()
    print(vehicle)

    # Opens the pickled ML-model
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    
    # Predicting values (in this case MPG)
    y_pred = predict_y(vehicle, model)

    # Returning the results
    result = {
        'y_pred (MPG)': list(y_pred)
    } 
    return jsonify(result)

# Local running of the Flask app
if __name__ == '__main__':
    
    df = load_data("Data-Projects/Web Apps/auto-mpg/model_files/auto-mpg.data")
    model = model_training(df)
    app.run(debug=True, host='127.0.0.1', port=5000)

