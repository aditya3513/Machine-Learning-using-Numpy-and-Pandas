import sys
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn import preprocessing

#list of column names for Data Frame
featureName = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
  "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "class"]

#reading CSV into a DataFrame
iris = pd.read_csv("housing.csv", names = featureName)

#get all columns except one with class names, for normalization
df = iris.loc[:, iris.columns != 'class']

#convert dataframe to list of lists
values = df.values

#normalize data values
min_max_scaler = preprocessing.MinMaxScaler()
values_scaled = min_max_scaler.fit_transform(values)

#normalized data frame
norm_df = pd.DataFrame(values_scaled)

#add class column 
norm_df["class"] = iris['class']

X_train = []
X_test = []
Y_train = []
Y_test = []

# for all k folds divide dataFrame in Test and Train
for k in range(10):
  msk = np.random.rand(len(norm_df)) < 0.8 #will divide the data frame randomly each time
  train = norm_df[msk] # train data (attributes)
  test = norm_df[~msk] #test data (class names)

  X_train.append(train.loc[:, iris.columns != 'class']) # training data, attribute values
  X_test.append(test.loc[:, iris.columns != 'class']) # test data, attribute values

  Y_train.append(train.loc[:, iris.columns == 'class']) # train data, class values
  Y_test.append(test.loc[:, iris.columns == 'class']) # test data, class values

'''
MSE: 
input: list of attributes (eg: sepal-width, sepal-width, sepal-lenth.......)
What it does: it calculates mean square error 
i.e probab of giving a wrong label
'''
#calculating the mean sqaured error
def MSE(attri_list):
    sum_list = 0
    for i in attri_list:
        sum_list += i
    se = [(x-sum_list)**2 for x in attri_list]
    #print(se)
    l = len(attri_list)
    if l == 0:
      l = 1
    mse = sum(se)/l
    return mse


'''
Dictionary: {key: [probability, "attribute-name"]}
this function returns the dictionary 
and the count for culumn or list of column count
dfX: Training Attribute DataFrame
dfY: Training Class Name DataFrame
'''
def createDictionary(dfX,dfY):
    #convert Dataframe to array for iterating as list
    yArray = dfY.values
    dictionary = {}
    k = 0

    for column,k  in zip(dfX, range(len(dfX))):
      attributeList = []
      for c, y in zip(dfX[column], yArray):
        attributeVal = []
        # c is the attribute for this Class
        attributeVal.append(c)
        # y[0] is the Class corresponding to attribute
        attributeVal.append(y[0])
        # structure: [probab, "attribute-name"]
        attributeList.append(attributeVal)
      dictionary[k] = attributeList
    return dictionary,dictionary.keys()

############################
#TREE Related stuff


"""
getBestFeatures: this function takes dictionary and returns the best feature 
  by comapring info gain with all other splits and takes one with max IG.
"""
def getBestFeatures(dictionary):
  best_info_gain = -1
  tempVar = []
  bestFeature = []

  #iterate over all element in dictionary and check if best feature.
  for ke in dictionary.keys():
    #get info gain and other values
    tempVar = meanSquareError(dictionary[ke])

    #if info gain more than previous gains, swap all values
    if tempVar[0] > best_info_gain:
        best_info_gain = tempVar[0]
        bestFeature = [tempVar[x] for x in range(1, len(tempVar))]
        bestFeature.insert(0, ke)

  return bestFeature

# def getAllAttBefore(sortedAttList, point):
#   beforeItem = []
#   for item in range(0, point):
#     beforeItem.append(sortedAttList[item][1])
#   return beforeItem


# def getAllAttAfter(sortedAttList, point):
#   AfterItem = []
#   for item in range(point, len(sortedAttList)):
#     AfterItem.append(sortedAttList[item][1])
#   return AfterItem


"""
MSE:
This function takes in list of attributes and gives 
1) MSE
2) mid point of split
3) split index for left child
4) split index for right child
"""
def meanSquareError(dictionalRow):
  #dictionalRow = [[probab, "class-name"], ........]

  #claculate MSE for parent 
  class_values = []
  #print(attri_list)
  for i in dictionalRow:
    class_values.append(i[1])
  mse_parent = MSE(class_values)
  
  max_mean_sqaure = 0
  maxAvg = 0
  bestSplitLeft = []
  bestSplitRight = []
  beforeIndex = [] #stores index of left list items
  afterIndex = [] #stores index of right list items
  beforeList = [] #stores class names of left list items
  afterList = [] #stores class names of left list items
  meanSquareErrorSplit = 0
  attListLen = float(len(dictionalRow))
  midPoints = []

# we sort the attribute list on basis of the probabilities
  sortedList = dictionalRow
  print("here")
#iterate over sorted array to find mid points where class names change
# as explained in example in class.

  # for i in range(len(sortedList)-1):
  #   item1 = sortedList[i]
  #   item2 = sortedList[i+1]
  #   m = 0
    
  #   if item1[1] != item2[1]:
  #     #calculate mid point
  #     m = (item1[0] + item2[0])/2
  #     #midPoints has all the mid points where i, i+1 had diff. class names.
  #     midPoints.append(m)


#for all mid points we find MSE. and select one with max MSE
  for i in range(len(sortedList)-1):
    item1 = sortedList[i]
    item2 = sortedList[i+1]
    m = (item1[0] + item2[0])/2
    for item in range(len(dictionalRow)):
      itemX = dictionalRow[item]
      
      #if this attribute is less than mid point then it goes to left list
      if itemX[0] <= m:
        beforeIndex.append(item)
        beforeList.append(itemX)
      else:
        #if this attribute is more than mid point then it goes to right list
        afterIndex.append(item)
        afterList.append(itemX)

    #MSE for left list
    mseLeft = MSE([x[1] for x in beforeList])
    #MSE for right list
    mseRight = MSE([x[1] for x in afterList])

    #weighted MSE Left list
    WEL = (mseLeft * (len(beforeList)/attListLen))
    #weighted Entropy Right list
    WER = (mseRight * (len(afterList)/attListLen))

    #mean Square Error 
    meanSquareErrorSplit = mse_parent - mseLeft - mseRight

    #if MSE is more than all previous swap all features
    if meanSquareErrorSplit > max_mean_sqaure:
      max_mean_sqaure = meanSquareErrorSplit
      maxAvg = m
      bestSplitRight = afterIndex
      bestSplitLeft = beforeIndex
  return max_mean_sqaure, maxAvg, bestSplitLeft, bestSplitRight 


      
def getDictAfterSplit(dictionary,index_split):
  dictAfterSplit = {}     
  classNames = []
  newDict = []
  for key in dictionary.keys():
    i = 0
    for item, i in zip(dictionary[key], range(len(dictionary[key]))):
      if i not in index_split:
        classNames.append(item[1])
        newDict.append(item)
    dictAfterSplit[key] = newDict

  return dictAfterSplit,classNames

"""
createDTree gives us the Binary Split Tree.
  Takes Dictionary of Training Dataset, class values of Training dataset,
  Stopping threshold.
"""
def createDTree(dictionaryDF, classValues, threshold):
  classValueList = []

  #check if type of data is DataFrame the convert to list
  if isinstance(classValues, pd.DataFrame):
    for c in classValues.values:
      classValueList.append(c[0])
  else:
    #check if type of data is not DataFrame then use as it is
    classValueList = classValues

  if len(set(classValues)) < threshold:
    s = 0
    for c in classValues.values.tolist():
      s += c[0]
    majority_val = s/float (len(classValues))
    root_node = majority_val,None,None,0,True
    return root_node

  else:
    #gets the best feature from the dictionar of dataset
    bestFeatures = getBestFeatures(dictionaryDF)
    #best Feature = [value of split, best mid value, left split indices, right split indices]
    print("got best feature")
    #gives dictionary for left Node after split on basis of best features
    LeftSplitData = getDictAfterSplit(dictionaryDF,bestFeatures[2])
    print("got left split feature")

    #gives dictionary for right Node after split on basis of best features
    RightSplitData = getDictAfterSplit(dictionaryDF,bestFeatures[3])
    print("got right split feature")
    
    #gives left Node after split on basis of best features, LeftSplitData
    leftchild = createDTree(LeftSplitData[0],LeftSplitData[1],threshold)
    print("got left child feature")
    
    #gives right Node after split on basis of best features, RightSplitData
    rightchild = createDTree(RightSplitData[0],RightSplitData[1],threshold)
    print("got right child feature")
    
    #gets root node based on best featuresm left and right node
    # last argument False since it as no leaf
    root_node = bestFeatures[0],rightchild,leftchild,bestFeatures[1],False
    print("got left split feature")
    return root_node

'''
Predict Class Name: takes test dataframe and root of tree
returns: the list of predicted values for each attribute value in test set
'''
def predictClassNames(TestDF, root):
    predictedValues = []

    #convert Dataframe to list
    Xlist = TestDF.values.tolist()

    #iterate over list to test each value and get prediction for them
    for testValue in Xlist:
      #get predicted values
      prectedClass = testData(testValue,root)

      #add to list of predicted values
      predictedValues.append(prectedClass)
    return predictedValues

'''
Accuracy tells us how much deviation is there in 
          predicted and actual values.
Returns: True Count, Wrong Count, Accuracy
'''
def Accuracy(TestDF , predicitedValues):

  #vallues to keep count, for confusion matrix  
  mse = 0

  # DataFrame to List Conversion for interation
  dfList = TestDF.values.tolist()

  # get list of class names from DataFrame List
  class_names = [x[0] for x in dfList]

  # count true, false and calculate accuracy
  for i in range(len(class_names)):
    error = class_names[i] - predicitedValues[i]
    mse += (error * error)
  return float(mse)/len(class_names)


'''
Test Data Runs all test Dataframe values through the Model and 
      predicts value.
Returns: Class Name that our model thinks is correct
'''
def testData(testValue,root):
  dictionaryTest ={}

  # create dictionary {index: Probability Value}
  for i in range(len(testValue)):
    dictionaryTest[i] = testValue[i]

  #root initialization and iterate to check the where to go in tree
  # based on closest mid point.

  node = root

  #root = [values, left, right, average mid point, isLeaf]

  while not node[4]: #checking if we have reached leaf.
      if dictionaryTest[node[0]] < node[3]: #if less than mid then left
          node = node[1]
      else: #if more than mid go right
          node = node[2]
  return node[0] #return the node value which is predicted(class Name)

"""
Run: this function runds the tree and prediction models.
takes threshold value and gives all k fiold result accuracy for each iteration
and average accuracy along with wrong, right count.
"""

def run(threshold):
  results = []
  avgAcc = 0
  #iterate for K folds
  for i in range(10):
    #get dictionary and features for the training dataFrame and class list
    dictionary, features = createDictionary(X_train[i], Y_train[i])

    #get Decision Tree using dictionary, class names and threshold
    DTree = createDTree(dictionary, Y_train[i], round(threshold * len(iris)))

    #prediction model based on the given tree
    prediction = predictClassNames(X_test[i], DTree)

    #resultsL right, wrong, accuracy on basis of test calss names and precition model
    acc = Accuracy(Y_test[i], prediction)
    print("\niteration ", i+1, "threshold = ", threshold)
    print("Accuracy = ", acc, "\n")
    print("-------------------------------------")

    avgAcc += acc
  return avgAcc/10

#stopping thresholds
thresholds = [0.05, 0.10, 0.15, 0.20]


avgAccList = []
#run the k fold for all values of threshold
for t in range(len(thresholds)):
  #get result and average accuracy
  averageAcc = run(thresholds[t])
  avgAccList.append(averageAcc)
df = pd.DataFrame({"threshold": thresholds, "average Accuracy": avgAccList})
print(df)

  
  
  

