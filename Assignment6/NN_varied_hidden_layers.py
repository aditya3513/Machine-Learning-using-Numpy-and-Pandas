import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import timeit


N = 8 # number of data inputs
H = 5 # number of hidden inputs
LR =  0.5 # learning rate
Max_iters = 10000 # max iterartions for convergence

#data matrix is just an identity matrix with 1's on diagonals
data = np.eye(N, dtype=float)

#input matrix
ipt = data

#sigmoid function as nonlinear function for tranformation
def sigmoid(X):
	
	sigmoid = (1.0 / (1.0 + np.exp(-X)))
	
	return sigmoid

def run(H):

	"""

	When iny input comes to an neuron its gonna go as 
	Input Xo and weights Wo and Bias b -> Xo * Wo  + b


	input layer -> hidden layer -> output layer

	then back propogate error and update biases

	this is the step that will be followed.

	input -> 8x8
	weights_input -> 8x3
	bias_input -> 1x3

	output -> 8x8
	weights_output -> 3x8
	bias_output -> 1x8
	"""

	#INITIALIZATION STEP
	#we set input weights with random values 
	#where rows are number of inputs and columns as Hidden inputs
	weight_input = np.random.random(size=(N,H))

	#now we will add bias for input layer, 
	#as we grom from input to hidden we just have 3 inputs so size will be [1x3]
	bias_input = np.ones(shape=(1, H))


	#we set output weight with random values
	#where rows are Hidden inputs and columns as number of inputs
	weights_output = np.random.random(size=(H,N))

	#now we will add bias for hidden layer, 
	#as we grom from input to hidden we just have 8 inputs so size will be [1x8]
	bias_hidden = np.ones(shape=(1, N))

	# print(weight_input.shape, weights_output.shape, bias_input.shape, bias_hidden.shape)

	# we run this algo this the weight converge or max iteration condition is met.
	for iters in range(Max_iters):
		# we perform this step for each row or each input.
		for rows_index in range(N):

			#input that is selected, this has all features from this input
			Xi = ipt[rows_index, :]
			Xi = Xi.reshape(1, N)
			#forwadring information using sigmoids,
			#first we send information from input to hidden, then from hidden to output

			#activation of hidden layer, ie from input to hidden.
			ha = sigmoid(np.dot(Xi, weight_input) + bias_input)

			#activation of output layer, ie from hidden to output.
			oa = sigmoid(np.dot(ha, weights_output) + bias_hidden)

			# then we back-propogate the error
			# from output to hidden and then hidden to input.

			#error in output layer
			error_output = (oa * ( (1-oa) * (Xi-oa) ) )

			#update weights by increasing weight by a delta value
			#delta value is but here we use product on error and input and add that to weights
			# so that weights are properly adjuested to input and the model
			#weights update for output layer
			#this works in hidden layer and error from output layer
			weights_output += LR * np.dot(ha.T, error_output)

			#error in hidden layer
			error_hidden = (ha * ( (1-ha) * np.dot(error_output, weights_output.T) ))

			#update -> LR * deltaW
			weight_input += LR * np.dot(Xi.T, error_hidden)

			# for i in range(H):
			bias_input = bias_input +  (LR * error_hidden)

			bias_hidden = bias_hidden +  (LR * error_output)



	"""
	now we generate the output matrix but applying sigmid and then 
		1) if value is less then 0.5 then 0 is predicted
		2) else if value if more then 0.5 then 1 is predicted
	"""
	output = np.empty(shape=ipt.shape)
	for i in range(N):
		Xi = ipt[i,:].reshape(1,N)
		#activation of hidden layer, ie from input to hidden.
		ha = sigmoid(np.dot(Xi, weight_input) + bias_input)

		#activation of output layer, ie from hidden to output.
		oa = sigmoid(np.dot(ha, weights_output) + bias_hidden)

		#if activation of output layer is more than 0.5 then predict 1 else 0
		#insert all values in putput array
		for j in range(N):
			if oa[:, j] > 0.5:
				output[i,j] = 1
			else:
				output[i,j] = 0
	acc = []
	for i in range(output.shape[0]):
		a = accuracy_score(output[i], ipt[i])
		acc.append(a)

	return np.mean(acc)

for i in range(1,10):

	print("-------------------- hidden layers = ", i)
	start = timeit.default_timer()
	print(run(i))
	stop = timeit.default_timer()

	print("time to run = ", stop - start)



