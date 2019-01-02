import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#reading data in nd-Arrays
data = np.genfromtxt ('perceptronData.csv', delimiter=",")

#shuffle data
np.random.shuffle(data)

# nd array of Features
X = data[:, :-1].astype(np.float)

#nd array of Labels
Y = data[:, -1].flatten()



#returns us the predicted values and runns it through sigmoid.
def predict(X, W):

	predictedValue = np.dot(X,W.T)

	return predictedValue

#returns updated weights after gradient descent
def gradient_descent(X, Y, W):



	converged = True

	while converged:

		converged = False
		# print(W)
		#predict values : 1000x4 * 4x1 = 1000x1
		predictions = predict(X, W)

		# Y -> 1x1000
		Y_temp = Y.T # 1000x1

		for i in range(X.shape[0]):
			value = Y_temp[i] * predictions[i]
			if value <= 0:
				W += Y_temp[i] * X[i, :]

		

	return W



def train(X, Y):

	W = np.zeros(X.shape[1]).reshape(1, X.shape[1])
	X = np.matrix(X)
	Y = np.matrix(Y)
	W = np.matrix(W)

	W = gradient_descent(X, Y, W)

	return W


def accuracy(X, Y, W):
	predictions = np.dot(X,W.T) #1000x4 * 4x1 = 1000x1

	correct = 0
	value = 0
	for i in range(len(predictions)):
		if predictions[i] >=0:
			value = 1
		else:
			value = -1

		if value == Y[i]:
			correct += 1

	total = X.shape[0] # number of samples
	acc = correct/ total #accuracy
	return acc



def run():

	#generating K folds, i.e 10 folds
	kf = KFold(n_splits=10)
	acc = []
	k = 0
	#this splis data in to 90:10 train-test split
	for train_index, test_index in kf.split(X):

		X_train, X_test = X[train_index], X[test_index]
		Y_train = Y[train_index].flatten()
		Y_test = Y[test_index].flatten()
		'''
		Y = [y1,y2,y3,y4.......,yn].T (nx1)
		'''

		scaler = StandardScaler().fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		X_train_scaled = np.insert(X_train_scaled, 0, np.ones(X_train.shape[0]), axis=1)

		X_test_scaled = np.insert(X_test_scaled, 0, np.ones(X_test.shape[0]), axis=1)
		'''
		X = [ [x1, x2, x3,......xm],
				.
				..................](nxm)
		'''
		
		W = train(X_train_scaled, Y_train)
		

		acc_x = accuracy(X_test_scaled, Y_test, W)
		acc.append(acc_x)

		print("accuracy = ", acc_x)
		k += 1

	print("Mean = ", np.mean(acc)*100)
	print("Stand Deviation = ", np.std(acc)*100)

run()



