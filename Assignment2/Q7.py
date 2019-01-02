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
data = pd.read_csv('concreteData.csv', sep=',')

def getSplitData(data, p):

	data = data.sample(frac=1)

	X = data.iloc[:, :-1]
	X = np.array(X,dtype=float)
	Y = data.iloc[:, -1]
	Xtemp = X
	if p != 1:
		for i in range(2, p+1):
			Xp = np.power(Xtemp, i)
			X = np.append(X, Xp, axis=1)

	X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size=0.1, random_state=42, shuffle=True)

	scalerX = StandardScaler(with_std=False).fit(X_train)
	X_train_scaled = scalerX.transform(X_train)
	X_test_scaled = scalerX.transform(X_test)


	y_train = np.matrix(y_train).T
	y_test = np.matrix(y_test).T

	scalerY = StandardScaler(with_std=False).fit(y_train)
	y_train = scalerY.transform(y_train)
	y_test = scalerY.transform(y_test)


	return X_train_scaled, X_test_scaled, y_train, y_test





def normalEquation(X, Y, lbd):

	# W = (Xt . X)^-1 Xt y

	# (Xt.X)
	xx = np.dot(X.T, X)

	#lambda * I
	li = (lbd * np.identity(xx.shape[0]))

	# Xt.X inverse
	xx_i = np.linalg.inv(xx + li)

	# (Xt.X + lbd * I)^-1 . Xt
	xx_i_xt = np.dot(xx_i, X.T)

	# (Xt.X + lbd * I)^-1 . Xt . Y
	W = np.dot(xx_i_xt, Y)
	
	return W

	# print(costs)


def rmse(Y, Y_pred):
	rmse = sqrt(mean_squared_error(Y, Y_pred))
	return rmse

def predictions(X, Y, W):
	return X.dot(W) + np.mean(Y)

def regression(data, p):
	Xtrain_list = []
	Xtest_list = []
	Ytrain_list = []
	Ytest_list = []
	lmd_val = np.arange(0.0, 500.0, 10)
	
	for k in range(10):
		xtr, xtst, ytr, ytst = getSplitData(data, p)
		
		Xtrain_list.append(xtr)
		Xtest_list.append(xtst)
		Ytrain_list.append(ytr)
		Ytest_list.append(ytst)

	RMSE_train_list = []
	RMSE_test_list = []

	for lmd in lmd_val:

		RMSE_train =[]
		RMSE_test = []
		W_train_list = []
		W_test_list = []

		for i in range(10):
			
			W_train_list.append( normalEquation(Xtrain_list[i], Ytrain_list[i], lmd) )

			W_test_list.append ( normalEquation(Xtest_list[i], Ytest_list[i], lmd))

			Pred_train = predictions(Xtrain_list[i], Ytrain_list[i], W_train_list[i])

			Pred_test = predictions(Xtest_list[i], Ytest_list[i], W_test_list[i])

			RMSE_train.append(rmse(Ytrain_list[i], Pred_train))
			RMSE_test.append(rmse(Ytest_list[i], Pred_test))

		print("Test RMSE  lambda  ", lmd, " = ",sum(RMSE_test)/float(len(RMSE_test)))
		print("Train RMSE lambda  ", lmd, " = ",sum(RMSE_train)/float(len(RMSE_train)))

		RMSE_train_list.append(sum(RMSE_train)/10)
		RMSE_test_list.append(sum(RMSE_test)/10)

	plotCurves(RMSE_train_list, RMSE_test_list, lmd_val)


def plotCurves(RMSE_train, RMSE_test, lmd):
	plt.plot(lmd, RMSE_train, label='RMSE Train')
	plt.plot(lmd, RMSE_test, label='RMSE Test')
	plt.xlabel("lambda")
	plt.ylabel("RMSE")
	plt.legend()
	plt.show()


regression(data, 5)











