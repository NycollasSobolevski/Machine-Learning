from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np

from joblib import load, dump

import pandas as pd

minNormalise = 0
maxNormalise = 0

def NormaliseFit(matrix, minValue=0, maxValue=1):
    global maxNormalise, minNormalise
    maxNormalise = np.max(matrix)
    minNormalise = np.min(matrix)
    

    numerator = np.array([x - minNormalise for x in np.nditer(matrix, order='C')])
    denominator = maxNormalise - minNormalise
    multiplier = maxValue - minValue
    
    XNorm = (numerator / denominator) * multiplier + minValue
    
    return XNorm

def NormaliseInput(data, minValue=0, maxValue=1):
    global maxNormalise, minNormalise

    numerator = np.array([x - minNormalise for x in np.nditer(data, order='C')])

    denominator = maxNormalise - minNormalise
    multiplier = maxValue - minValue
    
    XNorm = (numerator / denominator) * multiplier + minValue
    
    return XNorm


dataset = pd.read_csv("exoTrain.csv")
data = dataset.iloc[:,1:]
norm = NormaliseFit(data, 0, 1)
lines = dataset[dataset.columns[0]].count()
columns = len(data.columns)
X = norm.reshape((lines , columns))
X = pd.DataFrame(X)
y = dataset['LABEL']

def Fit():
    global X, y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=350
    )
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    model.fit(X,y)
    print(accuracy_score( y_test , model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))
    dump(model, "ExoplanetModel.pkl")

def Predict(data):
    model = load("ExoplanetModel.pkl")
    data = NormaliseInput(data)
    data = data.reshape((1,3197))
    data = pd.DataFrame(data)
    prediction = model.predict(data)
    return prediction

