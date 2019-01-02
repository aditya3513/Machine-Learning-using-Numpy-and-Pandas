import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
from copy import deepcopy

"""
steps:
1) Normalize data
2) get intitial centroids
3) calc dist of every feature from each centroid
4) recalculate centroid till diff is not less than tolerance
"""

"""
Setting up all parameter values:
"""
###################

file_name = "dermatologyData.csv"
#step 1 : gete data
#reading data in nd-Arrays
data = np.genfromtxt (file_name, delimiter=",")
# data = np.genfromtxt ('ecoliData.csv', delimiter=",")

#shuffle data
np.random.shuffle(data)

# nd array of Features
X = data[:, :-1].astype(np.float) #366 x 34

#nd array of Labels
Y = data[:, -1].flatten() # 366 X 1

K = len(set(Y))
max_iterations = 1000
tolerance = 0.1


# step 2: Normalizing data
scaler = StandardScaler()
scaler.fit(X)
scaler.transform(X) #now we have scaled X and is normalized

#step 3: initialize centroids with random values
def getCentroids(K):
	centroids = np.empty(shape=[K, X.shape[1]])
	for i in range(K):
		loc = random.randint(0,X.shape[0]-1)
		centroids[i] = X[loc]

	return centroids

def plotGraph(x,y, name):
    plt.figure()
    plt.ylabel('score')
    plt.xlabel('K')
    plt.plot(x, y)
    plt.savefig(name + ".png")

def hy_vals(Y):
	hy = 0
	total = len(Y)
	counts = dict(Counter(Y))
	for c in counts:
		p = counts[c]/total

		hy += (-1.0 *p) * math.log(p,2.0)

	return hy

def hc_values(z, Y):
	hc = 0
	total = len(Y)

	for i in range(z.shape[1]):
		p = list(z[:,i]).count(1) / total
		hc += (-1.0 *p) * math.log(p,2.0)

	return hc

def hyc_values(z,Y):
	hyc = []
	for j in range(z.shape[1]):
		pc = list(z[:,j]).count(1) / len(Y)
		# items in cluster j
		items_in_cluster = []
		for i in range(z.shape[0]):
			if z[i,j] == 1:
				items_in_cluster.append(Y[i])
		cluster_size = len(items_in_cluster)
		counts = dict(Counter(items_in_cluster))
		sum_hy = 0
		for c in counts:
			py = counts[c]/cluster_size
			sum_hy += py * math.log(py,2.0)
		val = (-1.0 * pc) * sum_hy
		hyc.append(val)

	return np.sum(hyc)


def calc_nmi(z, Y, hy):
	I = hy - hyc_values(z, Y)
	hc = hc_values(z, Y)

	nmi_score = (2.0 * I)/(hy + hc)
	return nmi_score

	
hy = hy_vals(Y)
sse_list = []
nmi_list = []
for temp in range(1,K):
	k = temp + 1
	# initialization
	centroids = getCentroids(k)

	
	obj_old = 0
	best_z = np.zeros(shape=[X.shape[0], k])
	for iters in range(max_iterations):
		J = 0
		z = np.zeros(shape=[X.shape[0], k])
		#create z matrix
		dist = np.zeros(shape=[X.shape[0], k])
		for i in range(X.shape[0]):
			
			for j in range(k):
				dist[i,j] = np.linalg.norm(X[i] - centroids[j], axis = 0)
			
			min_index = np.argmin(dist[i])
			z[i, min_index] = 1

		c_old = deepcopy(centroids)
		
		for i in range(k):
			count = 0
			points = []
			for j in range(1,X.shape[0]):
				if z[j,i] == 1:
					# points = np.vstack((points, X[j]))
					points.append(X[j])
					count += 1
			centroids[i] = np.mean(points, axis=0)

		
		
		error = np.linalg.norm(c_old - centroids, axis=0)
		if error.all() == 0:
			best_z = z
			break

	
	sse = 0
	for j in range(k):
		for i in range(X.shape[0]):
			sse += best_z[i,j] * np.linalg.norm(X[i] - centroids[j], axis = 0)

	nmi = calc_nmi(best_z, Y, hy)		

	print("--------------------")
	print("iteration = ",k)
	print("sse score = ", sse)
	print("nmi score = ", nmi)
	# print(k,sse)
	sse_list.append(sse)
	nmi_list.append(nmi)


x_axis = range(2,K+1)
plotGraph(x_axis, sse_list, "kmeans_sse_score_" + file_name)
plotGraph(x_axis, nmi_list, "kmeans_nmi_score_" + file_name)
			
