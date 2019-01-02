import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#reading data in nd-Arrays
data = np.genfromtxt ('spambase.csv', delimiter=",")

#shuffle data
np.random.shuffle(data)

# nd array of Features
X = data[:, :-1].astype(np.float)

#nd array of Labels
Y = data[:, -1].flatten()

for i in range(len(Y)):
    if Y[i] == 0:
        Y[i] = -1

max_iters = 10


scaler = StandardScaler()
scaler.fit(X)
scaler.transform(X) #now we have scaled X and is normalized

# each decision stump is just a feature threshold pair.
class DS():
    def __init__(self):

        #index of feature which was used to make this prediction
        self.feature = None

        #threshold for this feature
        self.T = 0.0

        #confidence about the prediction
        self.alpha = None

        self.label = 1



#this funcion takes un sorted unique feature set and returns thresholds,
#construct thesholds that are midway between successive feature values
# also adding first and last values too. like outliers
def calThreshold(unique_values):
    T = []
    #add value less than all values
    first = float(unique_values[0]) 
    T.append(first)
    for i in range(1,len(unique_values) -1):
        #add values which are in mid way
        mid = float(unique_values[i] + unique_values[i+1])/2.0
        T.append(mid)
    #add value more than all values
    last = unique_values[-1] + 0.5
    T.append(last)

    return T


def build_stump(X, Y, D):

    #this will store the best stump for all threshholds, the optimal one
    best_stump = DS()
    best_predictions = np.ones(X.shape[0])

    max_error = float("inf")

    for i in range(X.shape[1]):
        features = sorted(np.unique(X[:,i]))
        T = calThreshold(features)

        for t in T:
            #set all predictions to 1
            predictions = np.ones(X.shape[0])
            #error, all values are 
            # where ever there is misclassfication set value as -1
            predictions[X[:,i] < t] = -1
            
            error = np.sum(D[predictions != Y])
            error += 0.0001

            # if error > 0.5:
            #   error = 1 - error
            #   predictions = [p * -1.0 for p in predictions]

            
            if abs(0.5 - error) < max_error:
                max_error = error
                best_stump.T = t
                best_stump.feature = i
                best_predictions = predictions

    if max_error > 0.5:
        best_stump.label = -1


    return best_stump, best_predictions, max_error




def fit(X,Y):

    weak_clfs = []
    N = X.shape[0]
    D = np.full(N, 1/N)

    for iters in range(max_iters):
        best_stump, best_predictions, max_error = build_stump(X,Y,D)

        best_predictions = best_predictions * best_stump.label
        edge = 0
        for i in range(N):
            edge += D[i] * Y[i] * best_predictions[i]


        alpha = math.log((1.0 + edge)/(1.0 - edge))

        best_stump.alpha = alpha
        weak_clfs.append(best_stump)

        val = 0
        for i in range(len(Y)):
            val += Y[i] * best_predictions[i]

        exp_term = -1.0 * alpha * val
        print(alpha)
        D = (D * np.exp(exp_term))/np.sum(D)
        print("--------------------")
        print("T = ", best_stump.T)
        # print("predictions = ", best_predictions)
        print("accuracy_score = ", accuracy_score(Y, best_predictions))

    return weak_clfs


def predict(X,Y, clfs):
    predictions = np.zeros(X.shape[0])

    for clf in clfs:
        predictions[X[:,clf.feature] < clf.T] = -1
        predictions += clf.alpha * predictions


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clfs = fit(X_train,y_train)
predict(X_test,y_test,clfs)







