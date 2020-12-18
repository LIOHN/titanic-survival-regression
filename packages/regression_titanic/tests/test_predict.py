import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from regression_model.config import config
from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset
"""
def test1():
    assert 1;

"""
def test_make_single_prediction():
    # Given

    data = load_dataset(file_name= 'titanic.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here
    
    test_data = X_test.to_json(orient='records')
    pred = make_prediction(input_data=test_data)
    
    assert pred is not None
    assert accuracy_score(y_test, pred['predictions'].round()) < 1
