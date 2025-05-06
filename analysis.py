import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score

df = pd.read_csv('schizophrenia_dataset.csv')

# We mostly trained without exams scores to make our application more useful
# scaled_cols = ['Age', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score']
scaled_cols = ['Age']
categorical_cols = ['Marital_Status', 'Stress_Factors', 'Occupation']
unscaled_cols = ['Gender', 'Education_Level', 'Income_Level', 'Place_of_Residence', 'Family_History_of_Schizophrenia', 'Substance_Use', 'Social_Support', 'Medication_Adherence', 'Suicide_Attempt']

X = df[scaled_cols + categorical_cols + unscaled_cols]
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_to_scale = X_train[scaled_cols]
X_test_to_scale = X_test[scaled_cols]

scaler = StandardScaler()
X_train_scaled: NDArray[np.float64] = scaler.fit_transform(X_train_to_scale)
X_test_scaled: NDArray[np.float64] = scaler.transform(X_test_to_scale)

X_train_hotencode = pd.get_dummies(X_train[unscaled_cols + categorical_cols], columns=categorical_cols)
X_test_hotencode = pd.get_dummies(X_test[unscaled_cols + categorical_cols], columns=categorical_cols)

X_train_unscaled: NDArray[np.float64] = X_train_hotencode.to_numpy()
X_test_unscaled: NDArray[np.float64] = X_test_hotencode.to_numpy()

X_train_scaled_unscaled = np.concatenate((X_train_scaled, X_train_unscaled), axis=1)
X_test_scaled_unscaled = np.concatenate((X_test_scaled, X_test_unscaled), axis=1)

# Dummy classifier
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_scaled_unscaled, y_train)
dummy_prediction = dummy.predict(X_test_scaled_unscaled)
dummy_accuracy = accuracy_score(y_test, dummy_prediction)
print(f"Dummy score: {dummy_accuracy}")
precision = precision_score(y_test, dummy_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, dummy_prediction, average='macro')
print(f"Recall: {recall}")

# Perceptron
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=1)
perceptron.fit(X_train_scaled_unscaled, y_train)
perceptron_prediction = perceptron.predict(X_test_scaled_unscaled)
perceptron_accuracy = accuracy_score(y_test, perceptron_prediction)
print(f"Perceptron score: {perceptron_accuracy}")
precision = precision_score(y_test, perceptron_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, perceptron_prediction, average='macro')
print(f"Recall: {recall}")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled_unscaled, y_train)
lr_prediction = lr.predict(X_test_scaled_unscaled)
lr_accuracy = accuracy_score(y_test, lr_prediction)
print(f"Logistic Regression accuracy score: {lr_accuracy}")
precision = precision_score(y_test, lr_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, lr_prediction, average='macro')
print(f"Recall: {recall}")

# SVM

# Cross-validation
Cs = [0.001, 0.01, 0.1, 1]
scores = list()
for c in Cs:
    svm = SVC(kernel='linear', C=c)
    score = cross_val_score(svm, X_train_scaled_unscaled, y_train, cv=5).mean()
    print(f"Cross val. SVM (C={c}):", score)
    scores.append(score)

# sns.lineplot(x = Cs, y = scores, marker = 'o')
# plt.xlabel("C Values")
# plt.ylabel("Accuracy Score")
# plt.show()
c = Cs[scores.index(max(scores))]
print("Chosen C:", c)

svm = SVC(kernel='rbf', C=c)
svm.fit(X_train_scaled_unscaled, y_train)
svm_prediction = svm.predict(X_test_scaled_unscaled)
svm_accuracy = accuracy_score(y_test, svm_prediction)
print(f"SVM accuracy score: #{svm_accuracy}")
precision = precision_score(y_test, svm_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, svm_prediction, average='macro')
print(f"Recall: {recall}")

# Decision Tree

# Cross-validation
max_depths = list(range(1, 20, 2))
scores = list()
for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth)
    score = cross_val_score(dt, X_train_scaled_unscaled, y_train, cv=5).mean()
    print(f"Cross val. Decision Tree (max_depth={depth}):", score)
    scores.append(score)

# sns.lineplot(x = max_depths, y = scores, marker = 'o')
# plt.xlabel("Max Depth")
# plt.ylabel("Accuracy Score")
# plt.show()
max_depth = max_depths[scores.index(max(scores))]
print("Chosen max depth:", max_depth)

dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(X_train_scaled_unscaled, y_train)
dt_prediction = dt.predict(X_test_scaled_unscaled)
dt_accuracy = accuracy_score(y_test, dt_prediction)
print(f"Decision Tree accuracy score: {dt_accuracy}")
precision = precision_score(y_test, dt_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, dt_prediction, average='macro')
print(f"Recall: {recall}")

# k-NN

# Cross-validation
ks = list(range(1, 50, 10))
scores = list()
for n in ks:
    knn = KNeighborsClassifier(n_neighbors=n)
    score = cross_val_score(knn, X_train_scaled_unscaled, y_train, cv=5).mean()
    print(f"Cross val. kNN (n_neighbors={n}):", score)
    scores.append(score)

# sns.lineplot(x = ks, y = scores, marker = 'o')
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Score")
# plt.show()

k = ks[scores.index(max(scores))]
print("Chosen k:", k)

knn = KNeighborsClassifier(n_neighbors=k)
print("Cross val. kNN:", cross_val_score(knn, X_train_scaled_unscaled, y_train, cv=5).mean())
knn.fit(X_train_scaled_unscaled, y_train)
knn_prediction = knn.predict(X_test_scaled_unscaled)
knn_accuracy = accuracy_score(y_test, knn_prediction)
print(f"KNN accuracy score: {knn_accuracy}")
# print(f"KNN Score: #{knn.score(X_test_scaled_unscaled, y_test)}")
precision = precision_score(y_test, knn_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, knn_prediction, average='macro')
print(f"Recall: {recall}")

# MLP

# Cross-validation
alphas = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
scores = list()
for alpha in alphas:
    mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        hidden_layer_sizes=(22, 20, 20),
        random_state=1,
        alpha=alpha,
        early_stopping=True
    )
    score = cross_val_score(mlp, X_train_scaled_unscaled, y_train, cv=5).mean()
    print(f"Cross val. MLP (alpha={alpha}):", score)
    scores.append(score)

# sns.lineplot(x = alphas, y = scores, marker = 'o')
# plt.xlabel("Alpha Values")
# plt.ylabel("Accuracy Score")
# plt.show()

alpha = alphas[scores.index(max(scores))]
print("Chosen alpha:", alpha)

batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
scores = list()
for batch_size in batch_sizes:
    mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        hidden_layer_sizes=(22, 20, 20),
        random_state=1,
        alpha=alpha,
        early_stopping=True,
        batch_size=batch_size
    )
    score = cross_val_score(mlp, X_train_scaled_unscaled, y_train, cv=5).mean()
    print(f"Cross val. MLP (batch_size={batch_size}):", score)
    scores.append(score)

sns.lineplot(x = batch_sizes, y = scores, marker = 'o')
plt.xlabel("Batch Size")
plt.ylabel("Accuracy Score")
plt.show()
batch_size = batch_sizes[scores.index(max(scores))]
print("Chosen batch size:", batch_size)

mlp = MLPClassifier(
    activation='tanh',
    solver='adam',
    hidden_layer_sizes=(22, 20, 20),
    hidden_layer_sizes=(22, 20, 20),
    random_state=1,
    alpha=alpha,
    early_stopping=True,
    batch_size=batch_size,
)
mlp.fit(X_train_scaled_unscaled, y_train)
mlp_prediction = mlp.predict(X_test_scaled_unscaled)
mlp_accuracy = accuracy_score(y_test, mlp_prediction)
print(f"MLP Score: {mlp_accuracy}")
mlp = precision_score(y_test, mlp_prediction, average='macro')
print(f"Precision: {precision}")
recall = recall_score(y_test, mlp_prediction, average='macro')
print(f"Recall: {recall}")
