from helper import getData
from svm import svm_with_kernel
from dt import decisionTrees

def test():
    train_data, test_data = getData()
    svm_with_kernel(train_data[0], train_data[1], test_data[0], test_data[1])
    decisionTrees(train_data[0], train_data[1], test_data[0], test_data[1])

test()