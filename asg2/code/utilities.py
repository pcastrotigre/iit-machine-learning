import hashlib
import math

def repeatable_random(seed):
    hash = str(seed)
    while True:
        hash = hashlib.md5(hash).digest()
        for c in hash:
            yield ord(c)

def repeatable_shuffle(X, y):
    r = repeatable_random(42) 
    indices = sorted(range(X.shape[0]), key=lambda x: next(r))
    return X[indices], y[indices]

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def mapping_labels(labels):
	indexes = {}
	for i, label in enumerate(labels):
		indexes[i] = label
	return indexes	

def calculate_precision(confusion_matrix):
    return float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0])

def calculate_recall(confusion_matrix):
    return float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[0][1])

def calculate_accuracy(confusion_matrix):
    return float(confusion_matrix[0][0] + confusion_matrix[1][1]) / ( confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])

def calculate_fmeasure(precision,recall):
    return 2 * ( (precision * recall) / ( precision + recall ) )