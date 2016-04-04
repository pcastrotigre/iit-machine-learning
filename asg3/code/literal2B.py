from sklearn.cross_validation import KFold
from collections import defaultdict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_mldata
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

def get_hypothesis_softmax(thetas, Z):
    value = np.dot(Z,np.transpose(thetas))
    value = np.exp(value)

    return value/value.sum(axis=1)[:,None]

def get_hypothesis_sigmoid(thetas, Z):
    param = - np.dot(Z, np.transpose(thetas))
    return 1.00 / ( 1 + np.exp(param) )

def get_objective(k, Y_HAT, indicator):
    new_objective = 0.00
    for k_ in range(k):
        new_objective += np.sum(np.log(Y_HAT[:, k_][indicator[k_]]))
    return - new_objective

def gradient_descent(v_thetas, w_thetas, X, Y, max_iterations = 10000):
    """
        Get the best theta.
        It iterate over the theta until it finds that the theta does not change a lot (according to a threshold). It could finish when it finds the best
        theta or when it reaches a maximum of iterations. This in case our alpha is too small and we could come into a infinite loop.
        Return
            numpy array of the best theta
    """
    # constant to move in each step
    alpha = 0.0001

    # momentum
    momentum = 0.005

    # maximum difference that should be between the new theta and the theta to be considered as the best theta
    threshold = 0.001

    Z = get_hypothesis_sigmoid(w_thetas, X)
    Y_HAT = get_hypothesis_softmax(v_thetas, Z)

    k = v_thetas.shape[0]
    h = w_thetas.shape[0]

    # Indicator process to separate rows of one class from another
    indicator = defaultdict(list)
    for i, item in enumerate(Y):
        indicator[item].append(i)

    Y = Y.reshape(len(Y),1)

    objective = get_objective(k, Y_HAT, indicator)

    for i in range(max_iterations):

        for j in range(k): 
            
            h_ = Y_HAT[:,j]
            h_ = h_.reshape(len(h_),1)

            gradient = np.dot(np.transpose(Z),h_ - Y)

            # Calculate the new theta by multiplying the gradient with the constant alpha
            v_thetas[j] = v_thetas[j] - (alpha * np.transpose(gradient))

        for j in range(h):

            summ = 0.00
            for k_ in range(k):
                Y_ = Y_HAT[:,k_]
                Y_ = Y_.reshape(len(Y_),1)

                # Calculate the gradient
                y_ = np.zeros(len(Y))
                y_[indicator[k_]] = 1
                y_ = y_.reshape(len(y_), 1)

                substract = ( Y_ - y_ ) * v_thetas[k_][j]
                summ += np.sum(substract, axis = 0)

            # Calculate the gradient
            op1 = summ * Z[:,j] * ( 1 - Z[:,j] )
            op1 = op1.reshape(1,len(op1))
            gradient = np.dot(op1,X)

            # Calculate the new theta by multiplying the gradient with the constant alpha
            w_thetas[j] = w_thetas[j] - (alpha * (gradient)) + momentum

        Z = get_hypothesis_sigmoid(w_thetas, X)
        Y_HAT = get_hypothesis_softmax(v_thetas, Z)

        new_objective = get_objective(k, Y_HAT, indicator)
        if abs(objective - new_objective) < threshold:
            return v_thetas, w_thetas

        objective = new_objective

    return v_thetas, w_thetas

class LogisticRegressionKClasses:
    def fit(_self,x,y):
        '''
            Calculate all the parameters required to predict the data, such as the value of the parameters and the prior class
        '''
        h = 50

        # Per each class, calculate the value of the thetas zero
        w_thetas = np.array([ [0.01] * x.shape[1] ] * h)
        v_thetas = np.array([ [0.01] * h ] * len(set(y)))

        # Storing the data as a class variable  
        _self.v_thetas, _self.w_thetas = gradient_descent(v_thetas, w_thetas, x, y)

    def predict(_self, x):
        predictions_z = get_hypothesis_sigmoid(_self.w_thetas,x)
        predictions_y = get_hypothesis_softmax(_self.v_thetas, predictions_z)

        # To predict, for each example, take the class with the highest membership
        predicted = np.array([])
        for index in np.argmax(predictions_y, axis=1):
            predicted = np.append(predicted, index )

        return predicted

def logistic_regression(x,y):
    """
        Ierative multifeatures regression.
        Find the best theta by changing it in each iteration
        Print the training and testing errors for each mapping
    """
    errors_training_fmeasure = []
    errors_training_accuracy = []
    errors_testing_fmeasure = []
    errors_testing_accuracy = []

    regr = LogisticRegressionKClasses()

    poly = PolynomialFeatures(degree=1)

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

        # Predict over the testing data and getting the error
        predicted_y = regr.predict(x_2)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        precision, recall, f_measure, accuracy = get_measures(conf_matrix, len(set(y_train)))
        print 'Precision:', precision, ' Recall:', recall, ' Accuracy:', accuracy, ' F-Measure:', f_measure


mnist = fetch_mldata('MNIST original')
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

x, y = mnist.data / 255., mnist.target

# Shuffling the data
x, y = repeatable_shuffle(x, y)

# Logistic Regression
logistic_regression(x,y)