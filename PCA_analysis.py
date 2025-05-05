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

# scaled_cols = ['Age', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score']
scaled_cols = ['Age']
unscaled_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Level', 'Occupation', 'Place_of_Residence', 'Family_History_of_Schizophrenia', 'Substance_Use', 'Suicide_Attempt', 'Social_Support', 'Stress_Factors', 'Medication_Adherence']

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


# PCA
from PCA import PCA, reconstruct_error
ps = list(range(1,14))

for p in ps:
    all_data = np.concatenate((X_train_scaled_unscaled, X_test_scaled_unscaled), axis=0)
    pca_all = PCA(all_data, p)
    all_data_reduced = pca_all.get_reduced()
    all_reconstructed = pca_all.reconstruction(all_data_reduced)
    error = reconstruct_error(all_data, all_reconstructed)
    print('**** (All data) Reconstruction error for p = %d is %.4f ****' % (p, error))

    train_data_reduced = all_data_reduced[:X_train_scaled_unscaled.shape[0]]
    test_data_reduced = all_data_reduced[X_train_scaled_unscaled.shape[0]:]

    # pca_train = PCA(X_train_scaled_unscaled, p)
    # train_data_reduced = pca_train.get_reduced()
    # train_reconstructed = pca_train.reconstruction(train_data_reduced)
    # error = reconstruct_error(X_train_scaled_unscaled, train_reconstructed)
    # print('**** (Training data) Reconstruction error for p = %d is %.4f ****' % (p, error))

    # pca_test = PCA(X_test_scaled_unscaled, p)
    # test_data_reduced = pca_test.get_reduced()
    # test_reconstructed = pca_test.reconstruction(test_data_reduced)
    # error = reconstruct_error(X_test_scaled_unscaled, test_reconstructed)
    # print('**** (Test data) Reconstruction error for p = %d is %.4f ****' % (p, error))

    # Doing Logistic Regression on Reduced Data
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_data_reduced, y_train)
    lr_prediction = lr.predict(test_data_reduced)
    lr_accuracy = accuracy_score(y_test, lr_prediction)
    print(f"Logistic Regression accuracy score: #{lr_accuracy}")
    precision = precision_score(y_test, lr_prediction, average='macro')
    print(f"Precision: {precision}")
    recall = recall_score(y_test, lr_prediction, average='macro')
    print(f"Recall: {recall}")
    print("\n")