import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv('schizophrenia_dataset.csv')

scaled_cols = ['Age', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score']
unscaled_cols = ['Number_of_Hospitalizations', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Level', 'Occupation', 'Place_of_Residence', 'Family_History_of_Schizophrenia', 'Substance_Use', 'Suicide_Attempt', 'Social_Support', 'Stress_Factors', 'Medication_Adherence']

X = df[scaled_cols + unscaled_cols]
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_to_scale = X_train[scaled_cols]
X_test_to_scale = X_test[scaled_cols]

scaler = StandardScaler()
X_train_scaled: NDArray[np.float64] = scaler.fit_transform(X_train_to_scale)
X_test_scaled: NDArray[np.float64] = scaler.transform(X_test_to_scale)

X_train_unscaled: NDArray[np.float64] = X_train[unscaled_cols].to_numpy()
X_test_unscaled: NDArray[np.float64] = X_test[unscaled_cols].to_numpy()

X_train_scaled_unscaled = np.concatenate((X_train_scaled, X_train_unscaled), axis=1)
X_test_scaled_unscaled = np.concatenate((X_test_scaled, X_test_unscaled), axis=1)


# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled_unscaled, y_train)
lr_prediction = lr.predict(X_test_scaled_unscaled)
lr_accuracy = accuracy_score(y_test, lr_prediction)
print(f"Logistic Regression score: #{lr_accuracy}")
precision = precision_score(y_test, lr_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, lr_prediction, average='macro')
print(f"Recall: {recall}")

# SVM
svm = SVC(kernel='poly')
svm.fit(X_train_scaled_unscaled, y_train)
svm_prediction = svm.predict(X_test_scaled_unscaled)
svm_accuracy = accuracy_score(y_test, svm_prediction)
print(f"SVM score: #{svm_accuracy}")
precision = precision_score(y_test, svm_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, svm_prediction, average='macro')
print(f"Recall: {recall}")

# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(X_train_scaled_unscaled, y_train)
dt_prediction = dt.predict(X_test_scaled_unscaled)
dt_accuracy = accuracy_score(y_test, dt_prediction)
print(f"Decision Tree score: #{dt_accuracy}")
precision = precision_score(y_test, dt_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, dt_prediction, average='macro')
print(f"Recall: {recall}")

# k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled_unscaled, y_train)
knn_prediction = knn.predict(X_test_scaled_unscaled)
knn_accuracy = accuracy_score(y_test, knn_prediction)
print(f"KNN score: #{knn_accuracy}")
# print(f"KNN Score: #{knn.score(X_test_scaled_unscaled, y_test)}")

precision = precision_score(y_test, knn_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, knn_prediction, average='macro')
print(f"Recall: {recall}")
