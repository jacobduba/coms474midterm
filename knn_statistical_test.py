import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import itertools

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
perceptron_scores = []
lr_scores = []
svm_linear_scores = []
svm_poly_scores = []
svm_rbf_scores = []
dt_scores = []
mlp_scores = []

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

Cs = [0.001, 0.01, 0.1, 1]
c_scores = []
for c in Cs:
    lr = LogisticRegression(max_iter=1000, C=c)
    score = cross_val_score(lr, X_train_processed, y_full_train, cv=5).mean()
    c_scores.append(score)
optimal_c = Cs[np.argmax(c_scores)]
print(f"Optimal C found: {optimal_c}")

Cs = [0.001, 0.01, 0.1, 1]
c_scores = []
for c in Cs:
    svc = SVC(C=c, kernel='linear')
    score = cross_val_score(svc, X_train_processed, y_full_train, cv=5).mean()
    c_scores.append(score)
optimal_c_linear = Cs[np.argmax(c_scores)]
print(f"Optimal C for SVC (linear) found: {optimal_c_linear}")

Cs = [0.001, 0.01, 0.1, 1]
c_scores = []
for c in Cs:
    svc = SVC(C=c, kernel='poly')
    score = cross_val_score(svc, X_train_processed, y_full_train, cv=5).mean()
    c_scores.append(score)
optimal_c_poly = Cs[np.argmax(c_scores)]
print(f"Optimal C for SVC (linear) found: {optimal_c_poly}")

Cs = [0.001, 0.01, 0.1, 1]
c_scores = []
for c in Cs:
    svc = SVC(C=c, kernel='rbf')
    score = cross_val_score(svc, X_train_processed, y_full_train, cv=5).mean()
    c_scores.append(score)
optimal_c_rbf = Cs[np.argmax(c_scores)]
print(f"Optimal C for SVC (rbf) found: {optimal_c_rbf}")

max_depths = list(range(1, 20, 2))
max_depth_scores = []
for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    score = cross_val_score(dt, X_train_processed, y_full_train, cv=5).mean()
    max_depth_scores.append(score)
optimal_max_depth = max_depths[np.argmax(max_depth_scores)]
print(f"Optimal max depth for Decision Tree found: {optimal_max_depth}")

alphas = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
batch_sizes = [10, 20, 50, 100, 500, 1000, 2000]
alpha_and_batch = list(itertools.product(alphas, batch_sizes))
alpha_and_batch_scores = []
for alpha, batch_size in alpha_and_batch:
    mlp = mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        hidden_layer_sizes=(22, 20, 20),
        random_state=1,
        alpha=alpha,
        early_stopping=True,
        batch_size=batch_size
    )
    score = cross_val_score(mlp, X_train_processed, y_full_train, cv=5).mean()
    alpha_and_batch_scores.append(score)
optimal_alpha_and_batch = alpha_and_batch[np.argmax(alpha_and_batch_scores)]
optimal_alpha = optimal_alpha_and_batch[0]
optimal_batch_size = optimal_alpha_and_batch[1]
print(f"Optimal alpha and batch size for MLP found: {optimal_alpha}, {optimal_batch_size}")

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

    perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=1)
    perceptron.fit(X_train_processed, y_train)
    perceptron_pred = perceptron.predict(X_test_processed)
    perceptron_acc = accuracy_score(y_test, perceptron_pred)
    perceptron_scores.append(perceptron_acc)

    lr = LogisticRegression(max_iter=1000, C=optimal_c)
    lr.fit(X_train_processed, y_train)
    lr_pred = lr.predict(X_test_processed)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_scores.append(lr_acc)

    svm_linear = SVC(C=optimal_c_linear, kernel='linear')
    svm_linear.fit(X_train_processed, y_train)
    svm_linear_pred = svm_linear.predict(X_test_processed)
    svm_linear_acc = accuracy_score(y_test, svm_linear_pred)
    svm_linear_scores.append(svm_linear_acc)

    svm_poly = SVC(C=optimal_c_poly, kernel='poly')
    svm_poly.fit(X_train_processed, y_train)
    svm_poly_pred = svm_poly.predict(X_test_processed)
    svm_poly_acc = accuracy_score(y_test, svm_poly_pred)
    svm_poly_scores.append(svm_poly_acc)

    svm_rbf = SVC(C=optimal_c_rbf, kernel='rbf')
    svm_rbf.fit(X_train_processed, y_train)
    svm_rbf_pred = svm_rbf.predict(X_test_processed)
    svm_rbf_acc = accuracy_score(y_test, svm_rbf_pred)
    svm_rbf_scores.append(svm_rbf_acc)

    dt = DecisionTreeClassifier(max_depth=optimal_max_depth)
    dt.fit(X_train_processed, y_train)
    dt_pred = dt.predict(X_test_processed)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_scores.append(dt_acc)

    mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        hidden_layer_sizes=(22, 20, 20),
        random_state=1,
        alpha=optimal_alpha,
        early_stopping=True,
        batch_size=optimal_batch_size
    )
    mlp.fit(X_train_processed, y_train)
    mlp_pred = mlp.predict(X_test_processed)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_scores.append(mlp_acc)


    print(f"Trial {i+1}: KNN = {knn_acc:.4f}, Dummy = {dummy_acc:.4f}")

t_stat, p_value = stats.ttest_rel(knn_scores, dummy_scores, alternative='greater')
print("\nStatistical significance test (Paired t-test):")
print(f"KNN mean accuracy: {np.mean(knn_scores):.4f} ± {np.std(knn_scores):.4f}")
print(f"Dummy mean accuracy: {np.mean(dummy_scores):.4f} ± {np.std(dummy_scores):.4f}")
print(f"Mean difference: {np.mean(np.array(knn_scores) - np.array(dummy_scores)):.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.16f}")

if p_value < 0.05:
    print("Result: KNN is statistically significantly better than the dummy classifier (p < 0.05)")
else:
    print("Result: No statistically significant difference between KNN and dummy classifier (p >= 0.05)")

scores = dict()
scores['Dummy'] = dummy_scores
scores['KNN'] = knn_scores
scores['Perceptron'] = perceptron_scores
scores['Logistic Regression'] = lr_scores
scores['SVM (linear)'] = svm_linear_scores
scores['SVM (poly)'] = svm_poly_scores
scores['SVM (rbf)'] = svm_rbf_scores
scores['Decision Tree'] = dt_scores
scores['MLP'] = mlp_scores

# perform statistical significance tests of all classifiers against only dummy and perceptron
for key in scores.keys():
    for base in ['Dummy', 'Perceptron', 'KNN', 'Logistic Regression']:
        t_stat, p_value = stats.ttest_rel(scores[key], scores[base], alternative='greater')
        print(f"******* Statistical significance test (Paired t-test) between {key} and {base}: *******")
        print(f"{key} mean accuracy: {np.mean(scores[key]):.4f} ± {np.std(scores[key]):.4f}")
        print(f"{base} mean accuracy: {np.mean(scores[base]):.4f} ± {np.std(scores[base]):.4f}")
        print(f"Mean difference: {np.mean(np.array(scores[key]) - np.array(scores[base])):.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.16f}")
        print("")

        if p_value < 0.05:
            print(f"Result: {key} is statistically significantly better than {base} (p < 0.05)")
        else:
            print(f"Result: No statistically significant difference between {key} and {base} (p >= 0.05)")

        print("\n\n")

# Plotting the accuracies
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, n_trials + 1), knn_scores, label='KNN', marker='o')
# plt.plot(range(1, n_trials + 1), dummy_scores, label='Dummy', marker='o')
# plt.plot(range(1, n_trials + 1), perceptron_scores, label='Perceptron', marker='o')
# plt.plot(range(1, n_trials + 1), lr_scores, label='Logistic Regression', marker='o')
# plt.plot(range(1, n_trials + 1), svm_linear_scores, label='SVM (linear)', marker='o')
# plt.plot(range(1, n_trials + 1), svm_poly_scores, label='SVM (poly)', marker='o')
# plt.plot(range(1, n_trials + 1), svm_rbf_scores, label='SVM (rbf)', marker='o')
# plt.plot(range(1, n_trials + 1), dt_scores, label='Decision Tree', marker='o')
# plt.plot(range(1, n_trials + 1), mlp_scores, label='MLP', marker='o')
# # plt.ylim(0, 1)
# plt.xlabel('Trial')
# plt.ylabel('Accuracy')
# plt.title('Classifier Accuracies Over Trials')
# plt.xticks(range(1, n_trials + 1))
# plt.legend()
# plt.grid()
# plt.show()