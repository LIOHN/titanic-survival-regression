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

def test_make_predictions():
    data = load_dataset(file_name= config.DATASET_FILE)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # must be 0
    
    test_data = X_test
    original_data_length = len(test_data)
    pred = make_prediction(input_data=test_data)
    
    # Check predictions have been made
    assert pred is not None

    # Check accuracy
    assert accuracy_score(y_test, pred['predictions'].round()) < 1
    
    # Check number of predictions is the same as length of test data 
    # Change to != when more validation methods are added as some rows
    # may get removed.
    assert len(pred.get('predictions')) == original_data_length
