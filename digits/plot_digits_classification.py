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
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)
    
X_validate, X_test, y_validate, y_test = train_test_split(
    X_test, y_test, test_size=0.5, shuffle=False)

gamma_arr=[0.00001,0.0001,0.001,0.01,0.1,10]
for gamma_iter in gamma_arr:
	clf = svm.SVC(gamma=gamma_iter)
	clf.fit(X_train, y_train)
	predicted_train = clf.predict(X_train)
	
	#clf.fit(X_train, y_train)
	predicted_validate = clf.predict(X_validate)
	
	#clf.fit(X_train, y_train)
	predicted_test = clf.predict(X_test)
	print("Accuracy for", gamma_iter,"\t train is \t", metrics.accuracy_score(predicted_train,y_train),"\t\t\t validate is \t", metrics.accuracy_score(predicted_validate,y_validate),"\t test is \t", metrics.accuracy_score(predicted_test,y_test))








