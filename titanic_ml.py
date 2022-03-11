# This will run a final Evaluation and Test Using the RandomForestClassifier
# It will use the Best Parameter result in Model Tuning

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Reading the "Training DataSet"
training_attr = pd.read_csv('training_attr.csv')
training_surv_rslt = pd.read_csv('training_surv_rslt.csv')

# Reading the "Evaluation DataSet"
eval_attr = pd.read_csv('evaluation_attr.csv')
eval_surv_rslt = pd.read_csv('evaluation_surv_rslt.csv')

# Reading the "Test DataSet"
te_attr = pd.read_csv('test_attr.csv')
te_surv_rslt = pd.read_csv('test_surv_rslt.csv')

# Instance of our RandomForestClassifier
rf_eval = RandomForestClassifier(n_estimators=100, max_depth=20)
rf_eval.fit(training_attr, training_surv_rslt.values.ravel())

# We will try to check if the Model Score in the Field of Accuracy, Precision and Recall
# Scores will be from Eval DataSet and Test DataSet

eval_prdct = rf_eval.predict(eval_attr)
accuracy = round(accuracy_score(eval_surv_rslt, eval_prdct), 3)
precision = round(precision_score(eval_surv_rslt, eval_prdct), 3)
recall = round(recall_score(eval_surv_rslt, eval_prdct), 3)
print(f"Results: Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\n")

test_prdct = rf_eval.predict(te_attr)
accuracy_final = round(accuracy_score(te_surv_rslt, test_prdct), 3)
precision_final = round(precision_score(te_surv_rslt, test_prdct), 3)
recall_final = round(recall_score(te_surv_rslt, test_prdct), 3)
print(f"Results: Accuracy: {accuracy_final}\nPrecision: {precision_final}\nRecall: {recall_final}\n")
