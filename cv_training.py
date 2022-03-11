# Trying to check the score of the Model by Cross Validation using the Training DataSet
# It will Train on 80% of the Training DataSet then Test on the other 20%
# It will Iterate for 5 times and output the score

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Reading the "Training DataSet"
training_attr = pd.read_csv('training_attr.csv')
training_surv_rslt = pd.read_csv('training_surv_rslt.csv')

# Instance of our RandomForestClassifier
rf = RandomForestClassifier()

# 5-Fold Cross-Validation
cross_val_rslt = cross_val_score(rf, training_attr, training_surv_rslt.values.ravel(), cv=5)

# Printing the Training Score on Decimals
print(cross_val_rslt)