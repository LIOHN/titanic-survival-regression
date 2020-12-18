import pathlib

import regression_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

TRAINING_DATA_FILE = "titanic.csv"
# TESTING_DATA_FILE = "test.csv"
# TRAINING_DATA_FILE = "train.csv"

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "cabin", "embarked", "title"]
TARGET = "survived"

CATEGORICAL_VARS = ["sex", "cabin", "embarked", "title"]
NUMERICAL_VARS = ["age", "fare"]
CABIN = "cabin"

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ["age"]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]


PIPELINE_NAME = "logistic_regression.pkl"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05