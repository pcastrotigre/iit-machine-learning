import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve
from numpy.linalg import inv
from numpy.linalg import norm
from math import exp
from sklearn import linear_model
import tarfile
import pylab
import numpy as np


def show_and_save_data(filename, show_enable):
    f1 = open(filename , 'r')
    x = np.array([])
    y = np.array([])
    for line in f1:
        if not line.startswith('#'):
            row = line.split()
            x = np.append(x,float(row[0]))
            y = np.append(y,float(row[1]))
    if show_enable:
        plt.plot(x , y, 'bo')
        plt.title(filename)
        plt.show()  
    return x , y


#x, y = show_and_save_data('single_feature/svar-set1.dat',0)
#x, y = show_and_save_data('single_feature/svar-set2.dat',0)
#x, y = show_and_save_data('single_feature/svar-set3.dat',0)
#x, y = show_and_save_data('single_feature/svar-set4.dat',0)


def get_RSE_error(y_original,y_predicted):
    error = 0
    for i in range(0,len(y_original)):
        error = error + ( ((y_original[i]-y_predicted[i])**2)/(y_original[i]**2 ))
        
    SEE = error/len(y_original)
    return float(SEE)

def get_MSE_error(y_original,y_predicted):
    error = 0
    for i in range(0,len(y_original)):
        error = error +  ((y_original[i]-y_predicted[i])**2)
        
    SEE = error/len(y_original)
    return float(SEE)

#****************************************************************************************************************************************************************
#                                                    S I N G L E  F E A T U R E  :  L I N E A R  R E G R E S S I O N
#****************************************************************************************************************************************************************

class LinearRegression:
    """
    Class LinearRegression to fit and predict our model manually by multiplying matrices
    """
    def fit(_self,x,y):
        A = np.array([len(x),sum(x),sum(x),sum(np.power(x,2))])
        A = A.reshape((2,2))

        b = np.array([sum(y),np.dot(x,y)])
        b = b.reshape((2,1))

        _self.coefficients = solve(A,b)

    def predict(_self,x):
        return np.array([_self.coefficients[0] + _self.coefficients[1] * element for element in x])



def linear_regression_python(x,y):
    """
        Lineal Regression done with the python methods. 
        Use cross validation to get the RSE errors on testing and training data for each iteration of cross validation. 
        Then get the mean of the errors for training and testing 

        Return:
            numpy array with the mean of the training errors
            numpy array with the mean of the testing errors
    """
    regr = linear_model.LinearRegression()
    cv = KFold(len(x), n_folds=10)
    errors_test = np.array([])
    errors_training = np.array([])
    for train_idx, test_idx in cv:

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        x_train = x_train.reshape((len(x_train)),1)
        x_test = x_test.reshape((len(x_test)),1)

        regr.fit(x_train, y_train)

        # Predict over the training data and getting the error
        predicted_y_training = regr.predict(x_train)
        error_training = get_RSE_error(y_train,predicted_y_training)
        errors_training = np.append(errors_training,error_training)

        # Predict over the testing data and getting the error
        predicted_y = regr.predict(x_test)
        error_test = get_RSE_error(y_test,predicted_y)
        errors_test = np.append(errors_test,error_test)

    return np.mean(errors_training), np.mean(errors_test) 


def linear_regression_manual(x,y, plot_all = False):
    """
        Lineal Regression done manually by multiplying matrices. 
        Use cross validation to get the RSE errors on testing and training data for each iteration of cross validation. 
        Then get the mean of the errors for training and testing.
        If we want to plot the lineal model over the data, the parameter 'plot_all' must be set to True

        Return:
            numpy array with the mean of the training errors
            numpy array with the mean of the testing errors
    """
    model = LinearRegression()
    errors_test = np.array([])
    errors_training = np.array([])

    # Cross validation
    cv = KFold(len(x), n_folds=10)
    for train_idx, test_idx in cv:

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(x_train,y_train)

        # Predict over the training data and getting the error
        predicted_y_training = model.predict(x_train)
        error_training = get_RSE_error(y_train,predicted_y_training)
        errors_training  = np.append(errors_training,error_training)

        # Predict over the testing data and getting the error
        predicted_y = model.predict(x_test)
        error_test = get_RSE_error(y_test,predicted_y)
        errors_test  = np.append(errors_test,error_test)

    # Plot the model over all the data
    if plot_all:
        # Plot the data points
        plt.scatter(x, y, label="All points",color="blue")

        all_x = np.sort(x)
        all_x = all_x.reshape((len(all_x)),1)
        predicted_y = model.predict(all_x)
        
        # Plot the lineal model
        plt.plot(all_x, predicted_y, label="All points",color="red")
        plt.show()

    return np.mean(errors_training), np.mean(errors_test) 




#train_error, test_error = linear_regression_python(x,y)
#print 'Linear Regression: Python training error:',train_error
#print 'Linear Regression: Python testing error:',test_error

#train_error, test_error = linear_regression_manual(x,y,True)
#print 'Linear Regression: Manual training error:',train_error
#print 'Linear Regression: Manual testing error:',test_error



#****************************************************************************************************************************************************************
#                                                    S I N G L E  F E A T U R E  :  P O L Y N O M I A L  R E G R E S S I O N
#****************************************************************************************************************************************************************

def predict_polynomial(x, coefficients, poly):
    """
        Predict the values of x by multiplying the coefficients times the transpose of each Z
        Return 
            numpy array with the predicted values for each Z
    """
    predicted = np.array([])
    for element in x:
        Z = poly.fit_transform(element)
        predicted = np.append(predicted, np.dot(np.transpose(coefficients),np.transpose(Z)))
    return predicted


def polynomial_regression_manual(x,y, plot_all = False):
    """
        Polynomial Regression done manually by multiplying the matrices. 
        Use cross validation to get the RSE errors on testing and training data for each iteration of cross validation. 
        Then get the mean of the errors for training and testing.
        If we want to plot the polynomial model over the data, the parameter 'plot_all' must be set to True
    """
    if plot_all:
        plt.scatter(x, y, label="Dataset")
        all_x = np.sort(x)
        all_x = all_x.reshape((len(all_x)),1)

    for degree in [2,3,4,5,6]:
        errors_test = np.array([])
        errors_training = np.array([])

        # Initializing the degree of the polynomial
        poly = PolynomialFeatures(degree)

        # Cross validation
        cv = KFold(len(x), n_folds=10)
        for train_idx, test_idx in cv:

            x_train = x[train_idx]
            x_test = x[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Transforming the vectors to matrices
            x_train = x_train.reshape((len(x_train)),1)
            x_test = x_test.reshape((len(x_test)),1)

            # Get the Z for all our training x
            Z = poly.fit_transform(x_train)
            
            # Getting the coefficients by the dot product of the pinv of Z and the train Y vector
            coefficients = np.dot(np.linalg.pinv(Z),y_train)
            # Transforming the vector of coefficients into a matrix
            coefficients = coefficients.reshape((len(coefficients)),1)
            
            # Predict over the training data and getting the error
            predicted_y_training = predict_polynomial(x_train,coefficients,poly)
            error_training = get_RSE_error(y_train,predicted_y_training)
            errors_training  = np.append(errors_training,error_training)

            # Predict over the testing data and getting the error
            predicted_y = predict_polynomial(x_test,coefficients,poly)
            error_test = get_RSE_error(y_test,predicted_y)
            errors_test  = np.append(errors_test,error_test)    
        
        if plot_all:
            predicted_y = predict_polynomial(all_x,coefficients,poly)
            plt.plot(all_x, predicted_y, label="Degree %s" % degree)

        print 'Manual Polynomial error degree:',degree
        print 'Training error:',np.mean(errors_training)       
        print 'Testing error:',np.mean(errors_test)       

    if plot_all:
        plt.legend(loc='upper left')
        plt.show()


def polynomial_regression_python(x,y):
    """
        Polynomial Regression done with the python methods. 
        Use cross validation to get the RSE errors on testing and training data for each iteration of cross validation. 
        Then get the mean of the errors for training and testing.
    """
    for degree in [2,3,4,5,6]:
        errors_test = np.array([])
        errors_training = np.array([])

        # Initializing the degree of the polynomial
        poly = PolynomialFeatures(degree)

        # Cross validation
        cv = KFold(len(x), n_folds=10)
        for train_idx, test_idx in cv:

            x_train = x[train_idx]
            x_test = x[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Transforming the vectors to matrices
            x_train = x_train.reshape((len(x_train)),1)
            x_test = x_test.reshape((len(x_test)),1)

            # Transforming the train vector into the new dimension of the degree
            X_ = poly.fit_transform(x_train)
            clf = linear_model.LinearRegression()
            clf.fit(X_, y_train)

            # Predict over the training data and getting the error
            predicted_y_training = clf.predict(poly.fit_transform(x_train))
            error_training = get_RSE_error(y_train,predicted_y_training)
            errors_training  = np.append(errors_training,error_training)

            # Predict over the testing data and getting the error
            predicted_y = clf.predict(poly.fit_transform(x_test))
            error_test = get_RSE_error(y_test,predicted_y)
            errors_test  = np.append(errors_test,error_test)
            
        print 'Python Polynomial error degree:',degree
        print 'Training error:',np.mean(errors_training)       
        print 'Testing error:',np.mean(errors_test)     


#polynomial_regression_python(x,y)
#polynomial_regression_manual(x,y,True)




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
            row = line.split()
            features = []
            for i in range(len(row) - 1):
                features.append(float(row[i]))
            x.append(features)
            y.append(float(row[len(row)-1]))
    return np.array(x) , np.array(y)


#x, y = load_multiple_feature_data('multiple_features/mvar-set1.dat')
#x, y = load_multiple_feature_data('multiple_features/mvar-set2.dat')
x, y = load_multiple_feature_data('multiple_features/mvar-set3.dat')
#x, y = load_multiple_feature_data('multiple_features/mvar-set4.dat')

def predict_polynomial(x, coefficients, poly):
    predicted = np.array([])
    for element in x:
        Z = poly.fit_transform(element)
        predicted = np.append(predicted, np.dot(np.transpose(coefficients),np.transpose(Z)))
    return predicted

def primal_multifeature_regression(x,y):
    """
        Primal multifeature regression.
        Using cross validation to evaluate the performance with different mappings. 
        Print the training and testing error for each of the mappings
    """
    for degree in [2]:
        errors_test = np.array([])
        errors_training = np.array([])

        # Generating the polynomial
        poly = PolynomialFeatures(degree)

        # Cross validation
        cv = KFold(len(x), n_folds=10)
        for train_idx, test_idx in cv:

            x_train = x[train_idx]
            x_test = x[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Getting the matrix Z by transforming to a bigger degree
            Z_train = poly.fit_transform(x_train)

            # Getting the coefficients from the training data
            coefficients = np.dot(np.linalg.pinv(Z_train),y_train)

            # Predict over the training data and getting the error
            predicted_y = predict_polynomial(x_train, coefficients, poly)
            error_training = get_MSE_error(y_train,predicted_y)
            errors_training  = np.append(errors_training,error_training)

            # Predict over the testing data and getting the error
            predicted_y = predict_polynomial(x_test, coefficients, poly)
            error_test = get_MSE_error(y_test,predicted_y)
            errors_test  = np.append(errors_test,error_test)

        print 'Primal Multifeature Degree:',degree
        print 'Training error:',np.mean(errors_training)      
        print 'Testing error:',np.mean(errors_test)      


def gradient_descent(theta, Z, Y, max_iterations = 1000):
    """
        Get the best theta.
        It iterate over the theta until it finds that the theta does not change a lot (according to a threshold). It could finish when it finds the best
        theta or when it reaches a maximum of iterations. This in case our alpha is too small and we could come into a infinite loop.
        Return
            numpy array of the best theta
    """
    # constant to move in each step
    alpha = 0.000005

    # maximum difference that should be between the new theta and the theta to be considered as the best theta
    threshold = 0.001

    Y = Y.reshape(len(Y),1)
    # Getting the transpose of the Z
    Z_transpose = np.transpose(Z)

    for i in range(max_iterations):
        # Calculate the prediction
        hypothesis = np.dot(Z,theta)
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


def iterative_multifeature_regression(x,y):
    """
        Ierative multifeatures regression.
        Find the best theta by changing it in each iteration
        Print the training and testing errors for each mapping
    """
    for degree in [2]:
        errors_test = np.array([])
        errors_training = np.array([])

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
            
            # Setting the theta zero with random values from 0 to 1
            theta_zero = np.random.rand(Z_train.shape[1],1)

            # Get the best theta
            theta = gradient_descent(theta_zero, Z_train, y_train)
            
            # Predict over the training data and getting the error
            predicted_y = predict_polynomial(x_train, theta, poly)
            error_training = get_MSE_error(y_train,predicted_y)
            errors_training  = np.append(errors_training,error_training)

            # Predict over the testing data and getting the error
            predicted_y = predict_polynomial(x_test, theta, poly)
            error_test = get_MSE_error(y_test,predicted_y)
            errors_test  = np.append(errors_test,error_test)
            
        print 'Iterative Multifeature Degree:',degree
        print 'Training error:',np.mean(errors_training)      
        print 'Testing error:',np.mean(errors_test)    


def gaussian_kernel_function(x,y):
    sigma = 0.06
    numerator = pow(norm(x-y),2)
    denominator = 2 * pow(sigma,2)
    return exp(-(numerator/denominator))


def calculate_gramm_matrix(X):
    m,n = X.shape
    D = np.zeros((m,m))
    for i in range(m):
        print i
        for j in range(i,m):
            D[i,j] = gaussian_kernel_function(X[i],X[j])
            D[j,i] = D[i,j]
    return D


def predict_dual(alphas, x_train, x_test):
    transpose_alphas = np.transpose(alphas)
    predicted_y = np.zeros(len(x_test))
    for i, x in enumerate(x_test):
        prediction = np.zeros((len(x_train),1))
        for j, x_t in enumerate(x_train):
            prediction[j] = gaussian_kernel_function(x_t,x)
        predicted_y[i] = np.dot(transpose_alphas,prediction)
    return predicted_y


def dual_multifeature_regression(x,y):
    """
        Dual Multifeature Regression.
        By using the kernel gaussian function, calculate the alphas and the predictions
    """
    errors_test = np.array([])
    errors_training = np.array([])

    i = 0;
    # Cross Validation
    cv = KFold(len(x), n_folds=10)
    for train_idx, test_idx in cv:

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Calculating the Gramm matric
        G = calculate_gramm_matrix(x_train)

        # Get the alphas with the Gramm matrix and the Y
        alphas = np.dot(inv(G),y_train).reshape(len(y_train),1)
        
        # Predict over the testing data and getting the error
        predicted_y = predict_dual(alphas, x_train, x_test)
        error_test = get_MSE_error(y_test,predicted_y)
        errors_test  = np.append(errors_test,error_test)


    print 'Dual Multifeature Regression'
    #print 'Training error:',np.mean(errors_training)  
    print 'Testing error:',np.mean(errors_test)  



primal_multifeature_regression(x,y)
iterative_multifeature_regression(x,y)
#dual_multifeature_regression(x,y)



