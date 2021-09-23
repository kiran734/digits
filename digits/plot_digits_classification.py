"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

#print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from joblib import dump, load
from utils import preprocess, train, create_splits
import sys

scaling_factor = int(sys.argv[1])
train_size = int(sys.argv[2])
test_size = int(sys.argv[3])
validate_size = int(sys.argv[4])
	

digits = datasets.load_digits()
n_samples = len(digits.images)
data = preprocess(digits,n_samples,digits.images[0].shape,scaling_factor)

X_train, X_test, X_validate, y_train, y_test, y_validate = create_splits(data, digits.target, train_size, test_size,validate_size, shuffle=False)
    

gamma_arr=[0.00001,0.0001,0.001,0.01,0.1,10]
max_accuracy=0.11
best_gamma=0

for gamma_iter in gamma_arr:
	clf = svm.SVC(gamma=gamma_iter)
	train(clf,X_train, y_train)
	predicted_validate = clf.predict(X_validate)
	print("Accuracy for", gamma_iter,"\t\t\t validate is \t", metrics.accuracy_score(predicted_validate,y_validate))
	if metrics.accuracy_score(predicted_validate,y_validate) > 0.11:
		output_file="./models/tt_{}_val_{}_gamma_{}.joblib".format(
			test_size, validate_size,gamma_iter
		)
		dump(clf, output_file)
	if metrics.accuracy_score(predicted_validate,y_validate) > max_accuracy: 
		best_gamma=gamma_iter
		max_accuracy=metrics.accuracy_score(predicted_validate,y_validate)


output_file="./models/tt_{}_val_{}_gamma_{}.joblib".format(
			test_size, validate_size,best_gamma
		)

clf = load(output_file) 
predicted_test = clf.predict(X_test)
predicted_train = clf.predict(X_train)
test_accuracy = metrics.accuracy_score(predicted_test,y_test)
train_accuracy = metrics.accuracy_score(predicted_train,y_train)
print("The best gamma is: ", best_gamma," with validation accuracy: ",max_accuracy," train accuracy: ",train_accuracy, " test accuracy: ", test_accuracy)






