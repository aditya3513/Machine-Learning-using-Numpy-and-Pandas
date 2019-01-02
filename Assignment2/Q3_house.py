import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

learningRate = 0.0004
iterations = 1000
tolerance = 0.005
# featureNames = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# data = pd.read_csv('housing.csv', sep=',', names = featureNames)
data = pd.read_csv('housing.csv', sep=',')

def getSplitData(data):

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

def predictions(X, Y, W):
	return X.dot(W) + np.mean(Y)


def normalEquation(X, Y):

	# W = (Xt . X)^-1 Xt y

	# Xt.X
	xx = np.dot(X.T, X)

	# Xt.X inverse
	xx_i = np.linalg.inv(xx)

	# (Xt.X)^-1 . Xt
	xx_i_xt = np.dot(xx_i, X.T)

	W = np.dot(xx_i_xt, Y)
	
	return W

	# print(costs)


def regression():
	Xtrain_list = []
	Xtest_list = []
	Ytrain_list = []
	Ytest_list = []

	for k in range(10):
		xtr, xtst, ytr, ytst = getSplitData(data)
		
		Xtrain_list.append(xtr)
		Xtest_list.append(xtst)
		Ytrain_list.append(ytr)
		Ytest_list.append(ytst)

	RMSE_train_list = []
	RMSE_test_list = []

	for lmd in range(1):

		RMSE_train =[]
		RMSE_test = []
		W_train_list = []
		W_test_list = []

		for i in range(10):
			
			W_train_list.append( normalEquation(Xtrain_list[i], Ytrain_list[i]) )

			W_test_list.append ( normalEquation(Xtest_list[i], Ytest_list[i]))

			Pred_train = predictions(Xtrain_list[i], Ytrain_list[i], W_train_list[i])

			Pred_test = predictions(Xtest_list[i], Ytest_list[i], W_test_list[i])

			RMSE_train.append(rmse(Ytrain_list[i], Pred_train))
			RMSE_test.append(rmse(Ytest_list[i], Pred_test))

			print("Test RMSE  lambda  ", lmd, " = ",sum(RMSE_test)/float(len(RMSE_test)))
			print("Train RMSE lambda  ", lmd, " = ",sum(RMSE_train)/float(len(RMSE_train)))

			RMSE_train_list.append(sum(RMSE_train)/10)
			RMSE_test_list.append(sum(RMSE_test)/10)

	plotCurves(RMSE_train_list, RMSE_test, range(10))


def plotCurves(RMSE_train, RMSE_test, lmd):
	plt.plot(lmd, RMSE_train, label='RMSE Train')
	plt.plot(lmd, RMSE_test, label='RMSE Test')
	plt.xlabel("index")
	plt.ylabel("RMSE")
	plt.legend()
	plt.show()


regression()


# W_new = normalEquation(X_train_scaled, y_train)

# print(X_test_scaled.dot(W_new))






