import numpy as np
from sklearn.cross_validation import KFold
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from utilities import *


def read_file_multiple_feature(filename):
	'''
		Read file and save in memory the matrix with the features, and a vector with the labels. We take n features and 2 classes
	'''
	f1 = open(filename , 'r')
	x = []
	y = []
	for line in f1:
		row = line.split(',')
		features = []

		# Consider just the word counts (48 columns of data) to add them as features
		for i in range(len(row) - 10):
			row[i] = row[i].rstrip('\n')
			# Converting to binary
			if float(row[i]) > 0.00:
				features.append(1)
			else:
				features.append(0)

		# Adding to the features matrix X and labels vector Y
		x.append(features)
		row[len(row)-1] = row[len(row)-1].rstrip('\n')
		y.append(row[len(row)-1])
	return np.array(x) , np.array(y)



def calculate_prior_class(x, total):
	'''
		Calculate the prior class by selecting the number of examples of the class over the total number of examples
	'''
	return float(len(x))/total


def calculate_parameters(x):
	'''
		Calculate the parameters thetas. 
		Params:
			x: matrix of dimensions nxm where n is the number of examples (already applied the indicator) and m the number of features. Each [i,j] contains the word frequency
		Returns:
			Vector with dimension n, where n is the number of columns of X, to indicate the parameter for each feature
	'''
	e = 0.01
	return (np.sum(x, axis=0) + e ) / ( x.shape[0] + (2 * e) )

class NBBernoulliClassificationND:
	def fit(_self,x,y,labels):
		'''
			Calculate all the parameters required to predict the data, such as the value of the parameters and the prior class
		'''
		# Indicator process to separate rows of one class from another
		x_temp = defaultdict(list)
		for i, item in enumerate(y):
			x_temp[item].append(x[i])

		# Per each class, calculate the value of the parameters and the prior class 
		parameter = {}
		prior_class = {}
		for k, label in labels.items():
			x_tmp = np.array(x_temp[label])
			parameter[label] = calculate_parameters(x_tmp)
			prior_class[label] = calculate_prior_class(x_tmp,len(x))

		# Storing the data as a class variable	
		_self.parameters = parameter
		_self.prior_classes = prior_class

	def get_membership(_self,x,parameters,prior_class):
		'''
			Calculate the membership for each example. All of the parameters received belong to a single class
			Params:
				x: matrix that contains the examples of a specific class.
				parameters: vector that contains the parameter per each feature of a specific class
				prior_class: value with the prior_class of a specific class 
			Returns:
				Vector of nx1 where n is the number of examples of x. Each value is the membership of one example with respect of a specific class
		'''
		memberships = np.zeros(len(x))
		for i, item in enumerate(x):
			summ = 0.00
			for j, feature in enumerate(item):
				f1 = feature * math.log10(parameters[j])
				f2 = (1-feature) * math.log10(1-parameters[j])
				summ = summ + (f1+f2)
			memberships[i] = summ + math.log10(prior_class)
		return memberships

	def predict(_self,x, labels):
		'''
			Predict the class label for each example in x
		'''
		memberships = np.zeros(len(x))

		# For each class, calculate the membership of the examples
		for k, label in labels.items():
			membership = _self.get_membership(x,_self.parameters[label],_self.prior_classes[label])
			memberships = np.vstack((memberships, membership))

		# To predict, for each example, take the class with the highest membership
		predicted = np.array([])
		for index in np.argmax(memberships[1:,], axis=0):
			predicted = np.append(predicted, labels[index ])

		return predicted


def nb_bernoulli_nd_2classes(x,y):	
	regr = NBBernoulliClassificationND()
	cv = KFold(len(x), n_folds=10)
	errors_test = np.array([])
	errors_training = np.array([])
	for train_idx, test_idx in cv:

		x_train = x[train_idx]
		x_test = x[test_idx]
		y_train = y[train_idx]
		y_test = y[test_idx]

		labels = mapping_labels(np.unique(y_train))

		# Training
		regr.fit(x_train,y_train,labels)
		
		# Predict over the training data and getting the error
		predicted_y_training = regr.predict(x_train, labels)
		conf_matrix = confusion_matrix(y_train, predicted_y_training)
		precision = calculate_precision(conf_matrix)
		recall = calculate_recall(conf_matrix)
		accuracy = calculate_accuracy(conf_matrix)
		fmeasure = calculate_fmeasure(precision,recall)

		# Predict over the testing data and getting the error
		predicted_y_testing = regr.predict(x_test, labels)
		conf_matrix = confusion_matrix(y_test, predicted_y_testing)
		precision = calculate_precision(conf_matrix)
		recall = calculate_recall(conf_matrix)
		accuracy = calculate_accuracy(conf_matrix)
		fmeasure = calculate_fmeasure(precision,recall)

		print 'Precision:',precision, ' Recall:',recall, ' Accuracy:',accuracy,' F-Measure:',fmeasure


# Reading the data from the file and storing in memory
x, y = read_file_multiple_feature('../data/svar-set4.dat')

# Shuffling the data
x, y = repeatable_shuffle(x, y)

# Predicting the labels with the Naive Bayes Bernoulli
nb_bernoulli_nd_2classes(x,y)

