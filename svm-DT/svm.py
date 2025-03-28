from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm_with_kernel(train_label, train_data, test_label, test_data):
    svm = SVC(kernel='poly')

    svm.fit(train_data, train_label)

    prediction = svm.predict(test_data)

    accuracy = accuracy_score(test_label, prediction)

    print(accuracy)