from sklearn.svm import SVC

def svm_with_kernel(train_label, train_data, test_label, test_data):
    svm = SVC(kernel='poly')

    svm.fit(train_data, train_label)

    array = svm.predict(test_data)

    count = 0

    for i in range(len(test_label)):
        if array[i] == test_label[i]:
            count += 1
    
    print(count/len(test_label))