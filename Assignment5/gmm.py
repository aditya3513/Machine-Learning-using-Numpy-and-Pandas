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
from scipy.stats import multivariate_normal

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
K = 10
max_iterations = 100
tolerance = 0.1

###################

file_name = "yeastData.csv"
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
#step 2: Normalizing data
# scaler = StandardScaler()
# scaler.fit(X_feature)
# scaler.transform(X_feature) #now we have scaled X and is normalized

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

        hy += (-1.0 *p) * math.log(p,2)

    return hy

def hc_values(pi, Y):
    hc = 0
    total = len(Y)

    # for i in range(z.shape[1]):
    #     p = np.sum(z[:,i])/total
    #     hc += (-1.0 *p) * math.log(p,2)

    for i in range(len(pi)):
        hc += (-1 * pi[i]) * math.log(pi[i], 2)

    return hc

def hyc_values(z,Y, pi):
    hyc = []
    for j in range(len(pi)):
        pc = pi[j]
        # items in cluster j
        items_in_cluster = []

        sum_hy = 0
        items_in_cluster = []
        for i in range(z.shape[0]):
            if z[i,j] > 0:
                items_in_cluster.append(Y[i])
        cluster_size = len(items_in_cluster)
        counts = dict(Counter(items_in_cluster))
        sum_hy = 0
        for c in counts:
            py = counts[c]/cluster_size
            sum_hy += py * math.log(py,2.0)
        val = (-1.0 * pc) * sum_hy
        hyc.append(val)

        hyc.append(val)

    return np.sum(hyc)


def calc_nmi(z, Y, hy, pi):
    
    hc = hc_values(pi, Y)
   
    
    hyc = hyc_values(z, Y, pi)
    
    I = hy - hyc

    nmi_score = (2.0 * I)/(hy + hc)
    return nmi_score



def expectation(X,centroids, covar, pi, K):

    gamma = np.zeros(shape=[X.shape[0], K])

    for n in range(X.shape[0]):
        den = 0
        for j in range(K):
            
            den += pi[j] * multivariate_normal.pdf(X[n], mean=centroids[j], cov=covar[j], allow_singular=True)
        for k in range(K):
            
            num = pi[k] * multivariate_normal.pdf(X[n], mean=centroids[k], cov=covar[k], allow_singular=True)

            gamma[n,k] = num / den

    return gamma

def maximization(X,centroids, covar, pi, K, gamma):
    Nk = []
    for k in range(K):
        nk = np.sum(gamma[:,k], axis=0)
        Nk.append(nk)

    for k in range(K):
        mu_new = np.zeros(shape=[1,X.shape[1]])
        covar_new = np.zeros(shape=[X.shape[1],X.shape[1]])

        for n in range(X.shape[0]):
            mu_new += gamma[n,k] * X[n]
            
            val = X[n] - mu_new
            covar_new += gamma[n,k] * np.dot(val, val.T)

        centroids[k] = mu_new
        covar[k] = covar_new
        pi[k] = Nk[k]/X.shape[0]

    return centroids, covar, pi
        


    
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
            points = np.zeros(shape=[1, X.shape[1]])
            for j in range(1,X.shape[0]):
                if z[j,i] == 1:
                    if count == 0:
                        points = X[j]
                    else:
                        points = np.vstack((points, X[j]))
                    count += 1
            centroids[i] = np.mean(points, axis=0)
        
        
        error = np.linalg.norm(c_old - centroids, axis=0)
        best_z = z
        if error.all() == 0:
            break


    # till here we ran k means and we have got our initial centroids and Z matrix

    #INITIAL SET UP
    pi = []
    covar = []
    #initlize co variance:
    for j in range(k):
        p = 0
        total = len(best_z[:,j])
        points = np.zeros(shape=[1, X.shape[1]])
        for i in range(1,X.shape[0]):
            if best_z[i,j] == 1:
                if p == 0:
                    points = X[i]
                else:
                    points = np.vstack((points, X[i]))
                p += 1

        pi.append(p/total)
        covar.append(np.cov(points.T))


    # now we have initialized all the parameters
    # gamma = np.zeros(shape=[X.shape[0], k])

    #expectation step
    best_gamma = np.zeros(shape=[X.shape[0], k])
    ll_old = 0
    for iters in range(25):
        
        gamma = expectation(X,centroids, covar, pi, k)
        c_old = deepcopy(centroids)
        centroids, covar, pi = maximization(X,centroids, covar, pi, k, gamma)

        # error = np.linalg.norm(c_old - centroids, axis=0)
        # best_gamma = gamma
        # if error.all() < tolerance:
        #     break

        ll_new = 0.0
        for i in range(X.shape[0]):
            s = 0
            for j in range(k):
                s += pi[j] * multivariate_normal(mean=centroids[j], cov=covar[j], allow_singular=True).pdf(X[i])
            ll_new += np.log(s)
        if np.abs(ll_new - ll_old) < tolerance:
            best_gamma = gamma
            break
        ll_old = ll_new


    
    sse = 0
    for j in range(k):
        for i in range(X.shape[0]):
            sse += best_gamma[i,j] * np.linalg.norm(X[i] - centroids[j], axis = 0)

    nmi = calc_nmi(best_gamma, Y, hy, pi)     
    # nmi = sse
    print("--------------------")
    print("iteration = ",k)
    print("sse score = ", sse)
    print("nmi score = ", nmi)
    # print(k,sse)
    sse_list.append(sse)
    nmi_list.append(nmi)


x_axis = range(2,K+1)
plotGraph(x_axis, sse_list, "gmm_sse_score_" + file_name)
plotGraph(x_axis, nmi_list, "gmm_nmi_score_" + file_name)
            
