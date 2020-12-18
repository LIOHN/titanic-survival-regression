from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('titanic.csv')

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility
	
X_train.to_csv('train.csv', index=False)
X_test.to_csv('test.csv', index=False)