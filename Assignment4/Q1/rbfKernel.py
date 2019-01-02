import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#reading data in nd-Arrays
data = np.genfromtxt ('twoSpirals.csv', delimiter=",")

#shuffle data
np.random.shuffle(data)

# nd array of Features
X = data[:, :-1].astype(np.float)

#nd array of Labels
Y = data[:, -1].flatten()

iterations = 100


#returns us the predicted values and runns it through sigmoid.
def kernel(Xi, Xj):

	predictedValue = np.exp(-0.7 * np.linalg.norm(Xi - Xj) ** 2)

	return predictedValue

#returns updated weights after gradient descent
def dual_preceptron(X, Y, alpha):

	samples = X.shape[0]

	iteration = 0

	K = np.zeros((samples, samples))
	for i in range(samples):
		for j in range(samples):
			K[i,j] = kernel(X[i], X[j])

	while iteration < iterations:

		for i in range(samples):
			sum = 0

			for j in range(samples):
				Xprod = K[i,j] # (1x4) * (1000x4)T = 1x1000
				sum += alpha[j] * Y[:,j] * K[i,j]
				



			value = Y[:,i]*sum

			if value <= 0:
				alpha[i] += 1
				# print("added")


		iteration += 1
		print("completed = ", iteration*100/iterations)

	return alpha



def train(X, Y):

	alpha = np.zeros(X.shape[0]).flatten() #1x1000
	X = np.matrix(X) #1000x4
	Y = np.matrix(Y) #1x1000

	
	W = dual_preceptron(X, Y, alpha)

	return W

# print(train(X, Y))

# def accuracy(X, Y, W):
# 	predictions = np.dot(X,W.T) #1000x4 * 4x1 = 1000x1

# 	correct = 0
# 	value = 0
# 	for i in range(len(predictions)):

def accuracy(train_X,train_Y,test_X,test_Y,alpha):
    m = test_Y.size
    right = 0
    for i in range(m):
        s = 0
        for a, x_train,y_train  in zip(alpha, train_X,train_Y):
            s += a * y_train * kernel(test_X[i],x_train)
        if s >0:
            s = 1
        elif s <=0:
            s = -1
        if test_Y[i] == s:
            right +=1

    accuracy = right/test_X.shape[0]
    print(" Correct : ",right," Accuracy : ",accuracy)
    return accuracy
		







"""
if predictions[i] >=0:
			value = 1
		else:
			value = -1

		if value == Y[i]:
			correct += 1

	total = X.shape[0] # number of samples
	acc = correct/ total #accuracy
	return acc
"""



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
		

		acc_x = accuracy(X_train_scaled, Y_train, X_test_scaled, Y_test, W)
		acc.append(acc_x)

		print("accuracy = ", acc_x)
		k += 1

	print("Mean = ", np.mean(acc)*100)
	print("Stand Deviation = ", np.std(acc)*100)

run()



