from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
import numpy as np
from utilities import *

#****************************************************************************************************************************************************************
#                                                               M U L T I P L E  F E A T U R E S
#****************************************************************************************************************************************************************


def load_multiple_feature_data(filename):
    """
        Load the multiple feature dataset in memory. The last column in the file is always the regression
        Return
            numpy array matrix of the features
            numpy array of the labels
    """
    f1 = open(filename , 'r')
    x = []
    y = []
    for line in f1:
        if not line.startswith('#'):
            row = line.split(',')
            features = []
            for i in range(len(row) - 1):
                features.append(float(row[i]))
            x.append(features)
            label = row[len(row)-1]
            label = label.strip('\n')
            if label == 'Iris-setosa':
                y.append(1.00)
            else:
                y.append(0.00)

    return np.array(x) , np.array(y)

def predict(x, theta):
    hypothesis = get_hypothesis(theta, x)
    predicted = np.array([])
    for hypo in hypothesis:
        if hypo > 0.5:
            predicted = np.append(predicted,1)
        else:
            predicted = np.append(predicted,0)
    return predicted


def get_hypothesis(theta, Z):
    param = np.dot(Z,theta) * -1
    denominator = np.exp(param) + 1
    return 1.00 / denominator

def gradient_descent(theta, Z, Y, max_iterations = 1000):
    """
        Get the best theta.
        It iterate over the theta until it finds that the theta does not change a lot (according to a threshold). It could finish when it finds the best
        theta or when it reaches a maximum of iterations. This in case our alpha is too small and we could come into a infinite loop.
        Return
            numpy array of the best theta
    """
    # constant to move in each step
    alpha = 0.0005

    # maximum difference that should be between the new theta and the theta to be considered as the best theta
    threshold = 0.001

    Y = Y.reshape(len(Y),1)
    # Getting the transpose of the Z
    Z_transpose = np.transpose(Z)

    for i in range(max_iterations):
        # Calculate the prediction
        hypothesis = get_hypothesis(theta,Z)
        # Calculate the gradient
        gradient = np.dot(Z_transpose,hypothesis  - Y)
        # Calculate the new theta by multiplying the gradient with the constant alpha
        new_theta = theta - (alpha * gradient)
        # Calculate the difference between the new theta and the previous one
        difference = np.absolute(new_theta - theta)
        # If the difference is lesser than the threshold, return the new theta as the best theta, otherwise continue iterating
        if (np.sum(difference) < threshold):
            return new_theta
        theta = new_theta
 
    return theta

def logistic_regression(x,y):
    """
        Ierative multifeatures regression.
        Find the best theta by changing it in each iteration
        Print the training and testing errors for each mapping
    """
    for degree in [2]:

        # Generating the polynomial
        poly = PolynomialFeatures(degree)

        # Cross validation
        cv = KFold(len(x), n_folds=10)
        for train_idx, test_idx in cv:

            x_train = x[train_idx]
            x_test = x[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Mapping to a bigger dimension the training x data
            Z_train = poly.fit_transform(x_train)
            Z_test = poly.fit_transform(x_test)
            
            # Setting the theta zero with random values from 0 to 1
            theta_zero = np.random.rand(Z_train.shape[1],1)

            # Get the best theta
            theta = gradient_descent(theta_zero, Z_train, y_train)

            # Predict over the testing data and getting the error
            predicted_y = predict(Z_test,theta)
            conf_matrix = confusion_matrix(y_test, predicted_y)
            precision = calculate_precision(conf_matrix)
            recall = calculate_recall(conf_matrix)
            accuracy = calculate_accuracy(conf_matrix)
            f_measure = calculate_fmeasure(precision, recall)

            print 'Precision:', precision, ' Recall:', recall, ' Accuracy:', accuracy, ' F-Measure:', f_measure


# Load the data and save it in memory
x, y = load_multiple_feature_data('../data/iris1.dat')

# Shuffling the data
x, y = repeatable_shuffle(x, y)

# Logistic Regression
logistic_regression(x,y)