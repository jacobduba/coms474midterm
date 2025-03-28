from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def decisionTrees(train_label, train_data, test_label, test_data):

    clf = DecisionTreeClassifier()

    clf.fit(train_data, train_label)

    prediction = clf.predict(test_data)

    accuracy = accuracy_score(test_label, prediction)

    print(accuracy)