import pandas as pd
from sklearn.model_selection import train_test_split

# reading the CSV file to your DataFrame
csv_file = 'titanic_cleaned.csv'
titanic_df = pd.read_csv(csv_file)

# Spliting the Titanic Survive Attribute to the rest of the DataFrame
titanic_attr = titanic_df.drop('Survived', axis=1)
surv_rslt = titanic_df['Survived']

# Splitting of Training Data, Evaluation Data and Test Data
titanic_attr_tr, titanic_attr_test_eval, surv_rslt_tr, surv_rslt_test_eval = train_test_split(titanic_attr, surv_rslt, test_size=0.4, random_state=45)
titanic_attr_eval, titanic_attr_test, surv_rslt_eval, surv_rslt_test = train_test_split(titanic_attr_test_eval, surv_rslt_test_eval, test_size=0.5, random_state=45)

# Writing All the DataFrame to a CSV file
# 80% of DataSet for Training
titanic_attr_tr.to_csv('training_attr.csv', index=False)
surv_rslt_tr.to_csv('training_surv_rslt.csv', index=False)

# 20% of DataSet for Evaluation
titanic_attr_eval.to_csv('evaluation_attr.csv', index=False)
surv_rslt_eval.to_csv('evaluation_surv_rslt.csv', index=False)

# 20% of DataSet for Test
titanic_attr_test.to_csv('test_attr.csv', index=False)
surv_rslt_test.to_csv('test_surv_rslt.csv', index=False)
