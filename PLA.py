import numpy as np
import sys
from PIL import Image
import pandas as pd
import random
# from helper import *


def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE

    image_data1 = data[0]
    image_data2 = data[1]

    img1 = Image.fromarray(image_data1, mode="L")
    img2 = Image.fromarray(image_data2, mode="L")

    img1.save('../output/image_1.png')
    img2.save('../output/image_2.png')


    ### END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    ### YOUR CODE HERE

    ones_x = []
    ones_y = []
    fives_x = []
    fives_y = []

    for i in range(0, len(X)):
        if(y[i] == -1):
            ones_x.append(X[i][0])
            ones_y.append(X[i][1])
        else:
            fives_x.append(X[i][0])
            fives_y.append(X[i][1])

    fig, ax = plt.subplots(figsize=(10, 2.72))

    ax.plot(ones_x, ones_y, '*', label='ones', color='red')
    ax.plot(fives_x, fives_y, '+', label='fives')

    ax.legend()
    
    if(save):
        plt.savefig('../output/train_features.png')
    else:
        plt.show()


    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        
        n_samples, n_features = len(X), len(X[0])

        self.W = np.zeros(n_features)

        # print(n_samples , n_features)

        for iter in range(self.max_iter):
            if(iter % 100 == 0):
                # Print the iteration number to track progress
                print("Iteration #{}: Processing {} samples".format(iter+1, n_samples))
                
            for i in range(random.randint(0, n_samples), n_samples):
                # print("Iteration #{}: Processing sample {} of {}".format(iter+1, i+1, n_samples))
                xi = X[i]
                yi = y[i]
                prediction = self.predict(xi)
                if(prediction == yi):
                    continue
                error = yi
                self.W[1:] += error * xi[1:]
                self.W[0] += error              # bias
                break
            # print(self.W)

        # After implementation, assign your weights w to self as below:
        # self.W = w
        
        ### END YOUR CODE

        return self

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        try:
            Preds = []

            for xi in X:
                z = xi[1:] @ self.W[1:].T + + self.W[0]
                # print(z)

                if z >= 0: 
                    Preds.append(1)
                else:
                    Preds.append(-1)
            
            return Preds
        except:
            z = X[1:] @ self.W[1:].T + + self.W[0]
            # print(z)
            if z >= 0:
                return 1
            else:
                return -1


        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE

        predictions = self.predict(X)

        # print(predictions)

        n_samples = len(X)
        corrects = 0

        for pi, yi in zip(predictions, y):
            if pi == yi:
                corrects += 1
        
        return corrects / n_samples

        ### END YOUR CODE




def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    ### YOUR CODE HERE

    ones_x = []
    ones_y = []
    fives_x = []
    fives_y = []

    for i in range(0, len(X)):
        if(y[i] == -1):
            ones_x.append(X[i][0])
            ones_y.append(X[i][1])
        else:
            fives_x.append(X[i][0])
            fives_y.append(X[i][1])

    fig, ax = plt.subplots(figsize=(10, 2.7), layout='constrained')

    x = np.linspace(-1,0,100)

    ax.plot(ones_x, ones_y, '*', label='ones', color='red')
    ax.plot(fives_x, fives_y, '+', label='fives')
    ax.plot(x, x*(-W[1]/W[2]) - W[0]/W[2], label='Weights')

    ax.legend()
    
    plt.savefig('../output/result.png')

    ### END YOUR CODE



def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()

    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return W, train_acc, test_acc