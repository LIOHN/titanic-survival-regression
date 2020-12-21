from regression_model.config import config as model_config
from regression_model.processing.data_management import load_dataset
from regression_model import __version__ as _version

import json
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    response = flask_test_client.get('/health')

    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    response = flask_test_client.get('/version')

    assert response.status_code == 200
    response_json = json.loads(response.data)
    
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(file_name=model_config.DATASET_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        test_data.drop(model_config.TARGET, axis=1),
        test_data[model_config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here
    post_json = X_test.to_json(orient='records')

    response = flask_test_client.post('/v1/predict/regression',
                                      json=json.loads(post_json))

    # Check status code is OK
    assert response.status_code == 200
    
    # Check performance, accuracy.
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    assert prediction is not None
    assert accuracy_score(y_test, prediction) < 1
    
    # Check versions match
    response_version = response_json['version']
    assert response_version == _version

    # Check correct number of errors removed
    assert len(prediction) + len(
        response_json.get('errors')) == len(X_test)