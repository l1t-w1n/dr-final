import xgboost
from xgboost import XGBClassifier

class Estimator:
    @staticmethod
    def fit (X_train, y_train):
        return XGBClassifier().fit(X_train, y_train)
    
    @staticmethod
    def predict(model, X_test):
        return model.predict(X_test)