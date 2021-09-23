from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import numpy as np
def preprocess(digits,n_samples,shape,scaling_factor):
	new_data =np.zeros((n_samples,shape[0]*scaling_factor,shape[1]*scaling_factor))
	for i in range(0,n_samples):
		new_data[i]=rescale(digits.images[i], scaling_factor, anti_aliasing=True)
	return digits.images.reshape((n_samples, -1))
	
def train(clf,X,y):
	return clf.fit(X, y)
	
def create_splits(data,target,train_size,test_size,validation_size,shuffle):
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=(test_size+validation_size)/(validation_size+test_size+train_size), shuffle=False)
	X_validate, X_test, y_validate, y_test = train_test_split( X_test, y_test, test_size=validation_size/(validation_size+test_size), shuffle=False)
	return X_train,X_test,X_validate,y_train,y_test,y_validate
