import pandas as pd

df = pd.read_csv('schizophrenia_dataset.csv')

print(df.head())

# We can caluclate probabilities for all of these
# SVM (Platt scaling)
# Decision tree returns percent of classifiers at end
# Like k-NN gives the percentage of nearest neighbors.

# Logistic Regression

# SVM

svm = SVC(kernel='poly')
svm.fit(X_train_scaled_unscaled, y_train)
svm_prediction = svm.predict(X_test_scaled_unscaled)
svm_accuracy = accuracy_score(y_test, svm_prediction)
print(f"SVM score: #{svm_accuracy}")

# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(X_train_scaled_unscaled, y_train)
dt_prediction = clf.predict(X_test_scaled_unscaled)
dt_accuracy = accuracy_score(y_test, clf_prediction)
print(f"Decision Tree score: #{dt_accuracy}")

# k-NN
