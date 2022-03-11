# In this Cross Validation we are testing the best Hyperparameter
# Using the GridSearchCV, it will iterate for number of times and finding the best Result
# NOTE: We are still using the same Training DataSet

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Reading the "Training DataSet"
training_attr = pd.read_csv('training_attr.csv')
training_surv_rslt = pd.read_csv('training_surv_rslt.csv')

# Instance of our RandomForestClassifier
rf = RandomForestClassifier()

# We are declaring the number of parameters to check
parameters = {
    'n_estimators': [5, 50, 100],   # n_estimators is one of the parameter in RandomForestClassifier
    'max_depth': [2, 10, 20, None]  # max_depth is another
}

# GridSearchCV will use the specified parameters to train and test
# Multiple parameters are given to check which is best
cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(training_attr, training_surv_rslt.values.ravel())

# This will Print the best Combination of Parameter
print(cv.best_params_)
