from flask import Flask, request, jsonify
import pickle
from model_files.preprocessing import predict_y

# Initializing Flask app
app = Flask('app')

# Testroute
@app.route('/test', methods=['GET'])
def test():
    return 'Ping!'

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
# if __name__ == '__main__':
#     app.run(debug=True, host='127.0.0.1', port=5000)

