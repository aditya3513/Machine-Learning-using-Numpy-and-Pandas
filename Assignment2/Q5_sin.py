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

# data = pd.read_csv('housing.csv', sep=',', names = featureNames)
data = pd.read_csv('sinData_Train.csv', sep=',')
validationData = pd.read_csv('sinData_Validation.csv', sep=',')

def getSplitData(data, p):

	X = data.iloc[:, :-1]
	X = np.array(X,dtype=float)
	Y = data.iloc[:, -1]
	Xtemp = X
	if p != 1:
		for i in range(2, p+1):
			Xp = np.power(Xtemp, i)
			X = np.append(X, Xp, axis=1)

	X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

	return X, Y


def predictions(X, Y, W):
	return X.dot(W) + np.mean(Y)



def rmse(Y, Y_pred):
    rmse = sqrt(mean_squared_error(Y, Y_pred))
    return rmse


def normalEquation(X, Y):

	# W = (Xt . X)^-1 Xt y

	# Xt.X
	xx = np.dot(X.T, X)

	# Xt.X inverse
	xx_i = np.linalg.inv(xx)

	# (Xt.X)^-1 . Xt
	xx_i_xt = np.dot(xx_i, X.T)

	# (Xt.X)^-1 . Xt . Y
	W = np.dot(xx_i_xt, Y)
	
	return W

	# print(costs)

def regression(data, data_v, p):

	powers = range(1, p+1)

	RMSE_train = []
	RMSE_test = []

	for i in powers:

		Xtrain, Ytrain = getSplitData(data, i)

		Xtest, Ytest = getSplitData(validationData, i)

		W = normalEquation(Xtrain, Ytrain)

		RMSE_train.append( rmse(Ytrain, predictions(Xtrain, Ytrain, W)) )

		RMSE_test.append( rmse(Ytest, predictions(Xtest, Ytest, W)) )

		print("RMSE Train, power ", i, " = ", rmse(Ytrain, predictions(Xtrain, Ytrain, W)))
		print("RMSE Test, power ", i, " = ", rmse(Ytest, predictions(Xtest, Ytest, W)))


	plotCurves(RMSE_train, RMSE_test, powers)

def plotCurves(RMSE_train, RMSE_test, lmd):
	plt.plot(lmd, RMSE_train, label='RMSE Train')
	plt.plot(lmd, RMSE_test, label='RMSE Test')
	plt.xlabel("lambda")
	plt.ylabel("RMSE")
	plt.legend()
	plt.show()





regression(data, validationData, 65)






