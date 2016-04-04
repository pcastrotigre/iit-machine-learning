from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix
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
                y.append(0)
            elif label == 'Iris-versicolor':
                y.append(1)
            else:
                y.append(2)

    return np.array(x) , np.array(y)

def get_hypothesis(thetas, Z):
    value = np.dot(Z,np.transpose(thetas))
    value = np.exp(value)

    return value/value.sum(axis=1)[:,None]

def gradient_descent(thetas, Z, Y, max_iterations = 100000):
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

    k = set(y)

    # Indicator process to separate rows of one class from another
    indicator = defaultdict(list)
    for i, item in enumerate(Y):
        indicator[item].append(i)

    Y = Y.reshape(len(Y),1)

    for i in range(max_iterations):

        # Calculate the prediction
        hypothesis = get_hypothesis(thetas,Z)

        thetas_copy = np.copy(thetas)
        for j in range(len(k)):

            h_ = hypothesis[:, j]
            h_ = h_.reshape(len(h_), 1)

            # Calculate the indicator for the class
            y_ = np.zeros(len(Y))
            y_[indicator[j]] = 1.00
            y_ = y_.reshape(len(y_), 1)

            # Calculate the gradient
            gradient = np.dot(np.transpose(Z), h_ - y_)

            # Calculate the new theta by multiplying the gradient with the constant alpha
            thetas[j] = thetas[j] - (alpha * np.transpose(gradient))

        # Calculate the difference between the new theta and the previous one
        difference = np.absolute(thetas_copy - thetas)

        # If the difference is lesser than the threshold, return the new theta as the best theta, otherwise continue iterating
        if (np.sum(difference) < threshold):
            return thetas
    
    return thetas

class LogisticRegressionKClasses:
    def fit(_self,x,y):
        '''
            Calculate all the parameters required to predict the data, such as the value of the parameters
        '''
        # Per each class, calculate the value of the thetas zero
        thetas_zero = np.array([ [0.01] * x.shape[1] ] * len(set(y)))
        
        # Storing the data as a class variable  
        _self.thetas = gradient_descent(thetas_zero, x, y)

    def predict(_self, x):
        predictions = get_hypothesis(_self.thetas,x)

        # To predict, for each example, take the class with the softmax
        predicted = np.array([])
        for index in np.argmax(predictions, axis=1):
            predicted = np.append(predicted, index )

        return predicted

def logistic_regression(x,y):
    """
        Ierative multifeatures regression.
        Find the best theta by changing it in each iteration
        Print the training and testing errors for each mapping
    """
    regr = LogisticRegressionKClasses()

    poly = PolynomialFeatures(degree = 1)

    # Cross validation
    cv = KFold(len(x), n_folds=10)
    for train_idx, test_idx in cv:

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        x_ = poly.fit_transform(x_train)
        x_2 = poly.fit_transform(x_test)

        regr.fit(x_,y_train)

        # Predict over the testing data and getting the errors
        predicted_y = regr.predict(x_2)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        precision, recall, f_measure, accuracy = get_measures(conf_matrix, 3)
        print 'Precision:', precision, ' Recall:', recall, ' Accuracy:', accuracy, ' F-Measure:', f_measure


# Load the data and save it in memory
x, y = load_multiple_feature_data('../data/iris.dat')

# Shuffling the data
x, y = repeatable_shuffle(x, y)

# Logistic Regression
logistic_regression(x,y)