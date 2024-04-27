import json
import requests
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import DataLoader
from settings. constants import  VAL_CSV

with open('settings/specifications.json') as f:
    specifications = json.load(f)
    
info = specifications['description']
metrics = info['metrics']

test_df = pd.read_csv(VAL_CSV)

x_columns = specifications['description']['X']
y_columns = specifications['description']['y']

X_test_raw = test_df[x_columns]

data_loader = DataLoader()
data_loader.fit(X_test_raw)
X_test = data_loader.load_data()
y_test = X_test.RainTomorrow
print('X_test: ', X_test.head(10))

req_data = {'data': json.dumps(X_test_raw.to_dict())}
response = requests.get('http://127.0.0.1:8000/predict', data=req_data)
api_predict = response.json()['prediction']
print('predict: ', api_predict[:10])

api_score = eval(metrics)(y_test, api_predict)
print('accuracy: ', api_score)