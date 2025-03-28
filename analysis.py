import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('schizophrenia_dataset.csv')

X = df[['Age', 'Gender', 'Education_Level', 'Marital_Status', 'Occupation', 'Income_Level', 'Place_of_Residence', 'Family_History_of_Schizophrenia', 'Substance_Use', 'Suicide_Attempt', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score', 'Social_Support', 'Stress_Factors', 'Medication_Adherence', 'Medication_Adherence']]
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Perceptron (if need more models)

# Logistic Regression

# SVM

# Decision Tree (if need more models)

# k-NN
