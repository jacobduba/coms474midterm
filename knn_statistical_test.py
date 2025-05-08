import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('schizophrenia_dataset.csv')

scaled_cols = ['Age', 'Gender', 'Education_Level', 'Income_Level', 'Place_of_Residence',
              'Family_History_of_Schizophrenia', 'Substance_Use', 'Social_Support',
              'Medication_Adherence', 'Suicide_Attempt']
categorical_cols = ['Marital_Status', 'Stress_Factors', 'Occupation']
unscaled_cols = []

X = df[scaled_cols + categorical_cols + unscaled_cols]
y = df['Diagnosis']

n_trials = 10
dummy_scores = []
knn_scores = []

from sklearn.model_selection import cross_val_score

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_to_scale = X_full_train[scaled_cols]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_to_scale)

X_train_hotencode = pd.get_dummies(X_full_train[unscaled_cols + categorical_cols], columns=categorical_cols)
X_train_unscaled = X_train_hotencode.to_numpy()

X_train_processed = np.concatenate((X_train_scaled, X_train_unscaled), axis=1)

ks = list(range(1, 60))
k_scores = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_processed, y_full_train, cv=5).mean()
    k_scores.append(score)

optimal_k = ks[np.argmax(k_scores)]
print(f"Optimal k found: {optimal_k}")

for i in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i+42)

    X_train_to_scale = X_train[scaled_cols]
    X_test_to_scale = X_test[scaled_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_to_scale)
    X_test_scaled = scaler.transform(X_test_to_scale)

    X_train_hotencode = pd.get_dummies(X_train[unscaled_cols + categorical_cols], columns=categorical_cols)
    X_test_hotencode = pd.get_dummies(X_test[unscaled_cols + categorical_cols], columns=categorical_cols)

    X_train_unscaled = X_train_hotencode.to_numpy()
    X_test_unscaled = X_test_hotencode.to_numpy()

    X_train_processed = np.concatenate((X_train_scaled, X_train_unscaled), axis=1)
    X_test_processed = np.concatenate((X_test_scaled, X_test_unscaled), axis=1)

    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train_processed, y_train)
    knn_pred = knn.predict(X_test_processed)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_scores.append(knn_acc)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_processed, y_train)
    dummy_pred = dummy.predict(X_test_processed)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_scores.append(dummy_acc)

    print(f"Trial {i+1}: KNN = {knn_acc:.4f}, Dummy = {dummy_acc:.4f}")

t_stat, p_value = stats.ttest_rel(knn_scores, dummy_scores)
print("\nStatistical significance test (Paired t-test):")
print(f"KNN mean accuracy: {np.mean(knn_scores):.4f} ± {np.std(knn_scores):.4f}")
print(f"Dummy mean accuracy: {np.mean(dummy_scores):.4f} ± {np.std(dummy_scores):.4f}")
print(f"Mean difference: {np.mean(np.array(knn_scores) - np.array(dummy_scores)):.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.8f}")

if p_value < 0.05:
    print("Result: KNN is statistically significantly better than the dummy classifier (p < 0.05)")
else:
    print("Result: No statistically significant difference between KNN and dummy classifier (p >= 0.05)")
