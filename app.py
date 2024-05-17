from utils import Predictor
from utils import DataLoader
from pandas import DataFrame
from flask import Flask, request, jsonify, make_response

import pandas as pd
import json


app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    try:
        if request.method == 'GET':            
            received_keys = sorted(list(request.form.keys()))
            if len(received_keys) > 1 or 'data' not in received_keys:
                err = 'Wrong request keys'
                return make_response(jsonify(error=err), 400)

            data = json.loads(request.form.get(received_keys[0]))
            df = pd.DataFrame.from_dict(data)

            loader = DataLoader()
            loader.fit(df)
            processed_df = loader.load_data()
            print('processed: ', processed_df.columns)
            predictor = Predictor()
            processed_df = DataFrame(processed_df)
            prediction = predictor.predict(processed_df)
            response_dict = {'prediction': prediction.tolist()}

            return make_response(jsonify(response_dict), 200)
    except Exception as e:
        print('error: ', str(e))
        return make_response(jsonify(error=str(e)), 500)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000) 