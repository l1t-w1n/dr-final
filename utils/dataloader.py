from sklearn import preprocessing
import re
import numpy as np
from scipy import stats
import pandas as pd

class DataLoader(object):
    def fit(self, data):
        self.dataset = data.copy()
    
    def get_titles(self, name):
        pattern = '([A-Za-z]+)\.'
        title = re.search(pattern, name)
        if title:
            return title.group(1)
        return ''
    
    def load_data(self):          
        self.dataset = self.dataset.drop(columns=['Date','Temp9am', 'Temp3pm', 'Humidity9am'], axis=1)
        
        categorical_features = []
        numerical_features = []
        
        for col in self.dataset.columns:
            if self.dataset[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        #print(f"Categorical Features: {categorical_features}")
        #print(f"Numerical Features: {numerical_features}")
        for col in categorical_features:
            self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0])
            
        for col in numerical_features:
            self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mean())
                
        self.dataset['RainTomorrow'] = self.dataset['RainTomorrow'].map({'Yes': 1, 'No': 0})    
        self.dataset['RainToday'] = self.dataset['RainToday'].map({'Yes': 1, 'No': 0})
        
        le = preprocessing.LabelEncoder()
        self.dataset['WindDir9am'] = le.fit_transform(self.dataset['WindDir9am'])
        self.dataset['WindGustDir'] = le.fit_transform(self.dataset['WindGustDir'])
        self.dataset['WindDir3pm'] = le.fit_transform(self.dataset['WindDir3pm'])
        self.dataset['Location'] = le.fit_transform(self.dataset['Location'])

        self.dataset=self.dataset[(np.abs(stats.zscore(self.dataset)) < 3).all(axis=1)]
        
        return self.dataset

if __name__ == '__main__':
    df = pd.read_csv('./data/weatherAUS.csv')
    data_loader = DataLoader()
    data_loader.fit(df)
    data = data_loader.load_data()
    print(data.head())