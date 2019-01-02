import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.special import expit


#reading data in nd-Arrays
data = np.genfromtxt ('diabetes.csv', delimiter=",")

#shuffle data
np.random.shuffle(data)

# nd array of Features
X = data[:, :-1].astype(np.float)

#nd array of Labels
Y = data[:, -1].flatten()

#returns us the predicted values and runns it through sigmoid.
def predict(X, W):

	predictedValue = np.dot(X,W)
	
	'''
	Sigmoid function: 1.0/(1+e^-z)
	'''

	# sigmoid = (1.0 / (1 + np.exp(-predictedValue)))
	sigmoid = (1.0 / (1.0 + np.exp(-predictedValue)))

	return sigmoid

#returns Log loss for plotting against iterations
def logLoss(X, Y, W):

	predictions = predict(X,W)

	'''
	LL = ∑ Yi*(σ(Xi.W)) + (1 - Yi)*(1-σ(Xi.W))
	'''
	LL1 = np.dot(-Y.T, predictions)
	LL2 = np.dot((1 - Y.T), (1 - predictions))
	LL = LL1 - LL2

	#get avg loss for the whole feature set
	avgLL = LL.sum() / len(Y)

	return avgLL

#returns updated weights after gradient descent
def gradientDescent(X, Y, W, lr):

#predict values
	predictions = predict(X, W)
	

#calculate gradient by weighting the difference by Feature value
	gradient = np.dot(X.T, predictions - Y)

#update weight with learning rate

	W = W - (lr * gradient)

	return W

def train(X, Y, lr, iterations, showGraph, e):
	
	LLs = []
	iters_success = []
	W = np.zeros(X.shape[1]).reshape(X.shape[1], 1)

	for i in range(iterations):
		W = gradientDescent(X, Y, W, lr)
		LL = logLoss(X, Y, W)
		if len(LLs) > 1:
			# print(abs(LLs[-1] - LL))
			if abs(LLs[-1] - LL) < e:
				break

		iters_success.append(i)
		LLs.append(LL)

		if showGraph:
			plt.ylabel('epochs')
			plt.xlabel('Log Loss')
			plt.plot(LLs, iters_success)
			plt.savefig("LogLoss-Epochs.png")

		# if i % 50 == 0:
		# 	print(Wstar)

	return W, LLs


def accuracy(X, Y, W):
	predictions = np.dot(X,W)

	correct = 0
	value = 0
	for i in range(len(predictions)):
		if predictions[i] >=0.5:
			value = 1
		else:
			value = 0

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
		Y_train = Y[train_index].reshape(Y[train_index].shape[0], 1)
		Y_test = Y[test_index].reshape(Y[test_index].shape[0], 1)
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
		showGraph = False
		if k==9:
			showGraph = True
		W, LL = train(X_train_scaled, Y_train, 0.0004, 1000, showGraph, 0.0004)
		

		acc.append(accuracy(X_test_scaled, Y_test, W))

		k += 1

	print("Mean = ", np.mean(acc)*100)
	print("Stand Deviation = ", np.std(acc)*100)

run()



