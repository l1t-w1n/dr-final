import pickle
from settings.constants import SAVED_ESTIMATOR
from pandas import DataFrame

class Predictor:
    def __init__(self):
        self.loaded_estimator = pickle.load(open(SAVED_ESTIMATOR, 'rb'))

    def predict(self, data):
        print(f"data is a dataframe: {isinstance(data, DataFrame)}")
        data = DataFrame(data)
        prediction = self.loaded_estimator.predict(data)    
        print(f"prediction: {prediction}")
        return prediction