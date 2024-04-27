import pickle
import pandas as pd
from xgboost import XGBClassifier   
import json
from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV, VAL_CSV
from imblearn.over_sampling import SMOTE





with open('./settings/specifications.json') as f:
    specifications = json.load(f)
    
data = pd.read_csv(TRAIN_CSV)
X_train = specifications['description']['X']
y_train = specifications['description']['y']

X_raw = data[X_train]

data_loader = DataLoader()
data_loader.fit(X_raw)
X = data_loader.load_data()
y  = X.RainTomorrow
X = X.drop(columns=['RainTomorrow'], axis=1)

os = SMOTE()
X, y = os.fit_resample(X, y)

model = XGBClassifier(objective='binary:logistic')
model.fit(X, y)

with open('./models/XGBClassifier.pickle', 'wb') as f:
    pickle.dump(model, f)
    
raw_test = pd.read_csv(VAL_CSV)
X_test_raw = raw_test[X_train]
data_loader.fit(X_test_raw)
X_test = data_loader.load_data()
y_test = X_test.RainTomorrow
X_test = X_test.drop(columns=['RainTomorrow'], axis=1)
saved_model = pickle.load(open('./models/XGBClassifier.pickle', 'rb'))
predictions = saved_model.predict(X_test)
print(X_test.shape, y_test.shape)
print(f"Accuracy: {sum(predictions == y_test) / len(y_test) * 100}%")