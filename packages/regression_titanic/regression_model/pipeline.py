from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from regression_model.processing import preprocessors as pp
from regression_model.config import config
#from regression_model.processing import features

import logging

_logger = logging.getLogger(__name__)

titanic_pipe = Pipeline(
    [
        ('cabin_variable', pp.ExtractFirstLetter(variables=config.CABIN)),
            
        ('categorical_imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),

        ('missing indicator', pp.MissingIndicator(variables=config.NUMERICAL_VARS)),

        ('numerical_inputer', pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
         
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.05,
                variables=config.CATEGORICAL_VARS)),
         
        ('categorical_encoder', pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
         
        ('scaler', StandardScaler()),
        
        ('Linear_model', LogisticRegression(C=0.0005, random_state=0))
    ]
)
