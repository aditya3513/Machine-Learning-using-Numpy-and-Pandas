import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

learningRate = 0.001
iterations = 1000
tolerance = 0.001
# featureNames = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

data = pd.read_csv('yachtData.csv', sep=',')

def getSplitData():
	X = data.iloc[:, :-1]
	Y = data.iloc[:, -1]

	X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size=0.1, random_state=42)

	scaler = StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	X_train_scaled = np.insert(X_train_scaled, 0, np.ones(X_train.shape[0]), axis=1)

	X_test_scaled = np.insert(X_test_scaled, 0, np.ones(X_test.shape[0]), axis=1)

	y_train = np.matrix(y_train).T
	y_test = np.matrix(y_test).T


	return X_train_scaled, X_test_scaled, y_train, y_test

def rmse(Y, Y_pred):
    rmse = sqrt(mean_squared_error(Y, Y_pred))
    return rmse


def gradient_descent(X, Y, W):

	m = X.shape[0] # (# of rows)
	n = X.shape[1] # (columns or # of features)
	cost = 0
	costs = []
	Xt = X.T
	Rold = -1
	#iterating over given number of iteratons
	for i in range(iterations):
		#calculate Hypothesis value: i.e X.W
		H = np.dot(X, W.T)
		#calculate difference in actual and hypothesis values
		E = H - Y

		#calculating cost = (1/(2m)) * ∑(E)^2
		cost = (1/(2*m)) * np.sum(np.power(E,2))

		for k in range(n):
			W[0, k] -= (learningRate/(m)) * np.dot(E.T, X[:,k])

		Y_pred = np.dot(X, W.T).flatten()
		
			
		if (Rold - rmse(Y, Y_pred)) < tolerance and Rold != -1:
			print("tolerance met")
			break

		Rold = rmse(Y, Y_pred)

	costs.append(cost)
	return W, costs

	# print(costs)

def predictions(X, Y, W):
	return np.dot(X,W.T)

def regression(data, p):
	Xtrain_list = []
	Xtest_list = []
	Ytrain_list = []
	Ytest_list = []
	lmd_val = np.arange(0.0, 10.0, 0.2)
	
	for k in range(10):
		xtr, xtst, ytr, ytst = getSplitData()
		
		Xtrain_list.append(xtr)
		Xtest_list.append(xtst)
		Ytrain_list.append(ytr)
		Ytest_list.append(ytst)

	RMSE_train_list = []
	RMSE_test_list = []
	W = np.zeros(xtr.shape[1]).reshape(1, xtr.shape[1])

	RMSE_train =[]
	RMSE_test = []
	W_train_list = []
	W_test_list = []

	for i in range(10):

		W_train_list.append(gradient_descent(Xtrain_list[i], Ytrain_list[i], W))
		pred = predictions(Xtrain_list[i], Ytrain_list[i], W)
		RMSE_train.append( rmse(Ytrain_list[i], pred) )
		predTest = predictions(Xtest_list[i], Ytest_list[i], W)
		RMSE_test.append( rmse( Ytest_list[i], predTest) )
			# 	RMSE_test.append(rmse(Ytest_list[i], Pred_test))

		print("Test RMSE  iterations  ", i, " = ",sum(RMSE_test)/float(len(RMSE_test)))
		print("Train RMSE iterations  ", i, " = ",sum(RMSE_train)/float(len(RMSE_train)))

		RMSE_train_list.append(sum(RMSE_train)/10)
		RMSE_test_list.append(sum(RMSE_test)/10)

	plotCurves(RMSE_train_list, RMSE_test_list, range(10))


def plotCurves(RMSE_train, RMSE_test, lmd):
	plt.plot(lmd, RMSE_train, label='RMSE Train')
	plt.plot(lmd, RMSE_test, label='RMSE Test')
	plt.xlabel("lambda")
	plt.ylabel("RMSE")
	plt.legend()
	plt.show()


regression(data, 5)





