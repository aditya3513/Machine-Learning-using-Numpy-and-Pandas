import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

"""
This function creates a list or arrays where,
each array having elemnent in earch row of the file
"""
def readFile(fileName):
	#list to store each line
	content = []
	#read given file name
	with open(fileName, "r") as file:
		#iterate over each file
		for line in file:
			#split converts it to array and adds it to content[]
			content.append(line.split())

	return content

X_train = readFile("train.data")
Y_train = readFile("train.label")

X_test = readFile("test.data")
Y_test = readFile("test.label")

"""
This function creates a dictionary with elements lik: {'wordId': total_count_in_corpus}
"""
def createVocab():
	#vocab to store all words, key is wordId and value is total count in corpus
	vocabX = {}
	# for every line we get in file here train.data
	for line in X_train:
		
		#since train.data has format: docId wordId count, using that below
		docId = line[0]
		wordId = line[1]
		count = line[2]

		#if we find that word in vocab then update count
		if wordId in vocabX:
			vocabX[wordId] += int(count)
		else: #else add the word to vocab.
			vocabX[wordId] = 1

	return vocabX 


#since first question says we have to sort the aray by frequency so this function does that.
def sortVocab(vocab):
	#sorted function returns the sorted array and reverse=True haves it descending.
	Freq = sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)
	return Freq

"""
Since prior prob is log( (prob of x happening)/(total prob) )
we first calculate the occurance or each label or class and add it to a dict,
then we sum all values up to get total freq
then we canculate prior prob for each label.
"""
def priorProb():

	#this stores class labels as keys and prior probs as values
	priorProbDict = {}

	#this stores class label as key and count as values.
	class_dict = {}
	for line in Y_train:

		label = line[0]
		#check if key is in class_dict, if yes then update count
		if label in class_dict:
			class_dict[label] += 1
		else: #else add it to dict with count =1
			class_dict[label] = 1

	#now we calculate sum of all values to get total sum for probab claculation.

	totalProb = sum(class_dict.values())

	for label in class_dict.keys():
		prob = class_dict[label]/totalProb
		priorProbDict[label] = np.log(prob)

	return priorProbDict, class_dict


'''
Since each label in train.lebl or Y_train | test.label or Y_test represents a document
so this function creates a mapping dict, where key is document_number and value is label
returns docClassDict
'''
def createDocClassDict(Y):
	# we keep a count of all documents as use it as key for docClassDict
	i = 1
	docClassDict = {}
	for line in Y:
		label = line[0]

		docClassDict[str(i)] = label
		i += 1

	return docClassDict

"""
here we create a dictionary of structure like this:
{
	docId: {
		wordId: count, ......
	}, ........
}

we get frequency of each word in each document in form of a dictionary
"""
def createTestDict():

	test_dict = {} # as mentioned above
	# wordFreqDict = {} # dict which has structure: key as wordId, values as count

	for line in X_test:
		
		docId = line[0]
		wordId = line[1]
		count = line[2]

		# we first check if the document has this particular document
		if docId in test_dict:
			
			#then we check if this document has this word
			if wordId in test_dict[docId]:
				test_dict[docId][wordId] += int(count)

			#else, we dont find this word in this doc then we add it to it
			else:
				test_dict[docId][wordId] = int(count)
		#else, we dont find this doc, then we add this word and its cunt to this doc.
		else:
			#temp sice right now we cannot do chain access as key is not present
			temp = {}
			#first make child and then add to parent to create nested dict.
			temp[wordId] = int(count)
			test_dict[docId] = temp

	return test_dict

'''
This helps us to create a bernoulli dist and link train.data which are stored into a vocab 
be lnked with docClassList which is made from Y_train ot train.label.
count of classes will be 20 for a 20 news group and all this is stored into one class list
'''
def BernoulliDist(vocab):

	docClassDict = createDocClassDict(Y_train)
	classList = {str(k): {} for k in range(1, 21, 1)}
	
	for line in X_train:
		
		docId = line[0]
		wordId = line[1]
		count = line[2]


		if wordId in vocab:

			classKey = docClassDict[docId]
			if wordId in classList[classKey]:
				classList[classKey][wordId] += 1
			else:
				classList[classKey][wordId] = 1

	return classList

"""
this function calculates conditional probabilities of all words across all documents.
it uses classDict that is claculated in Prior Probab and used for claculations, 
as it stores how many times each word apperaed in a document.
"""
def ConditionalProbDict(vocab, classDict, classList):

	condProbClassDict = {str(k): {} for k in range(1, 21, 1)}
	for class_i in classList:

		documentCount = classDict[class_i]
		condProbDict = {}
		for wordId in vocab:

			if wordId in class_i:

				word_count = classList[class_i][wordId]
				prob = (word_count + 1) / float(documentCount + 2)
				condProbDict[wordId] = prob
			else:
				# in this case prob of finding word in class is zero so Numerator is (0 + 1) = 1
				prob = (0 + 1) / float(documentCount + 2)
				condProbDict[wordId] = 	prob	

		condProbClassDict[class_i] = condProbDict

	return condProbClassDict


def predict(vocab,Vsize, condProbClassDict, priorProbDict):

	predictionsDict = {}
	for test_docId in test_dict:
		maxScore = -float("inf")
		bestClass = 0

		#  if word in class then add score 
		for classId in condProbClassDict:
			score  = priorProbDict[classId]

			for wordId in vocab:

				if wordId in test_dict[test_docId]:
					score += np.log(condProbClassDict[classId][wordId])
				else: # not found case so subtract from 1
					score += np.log(1 - condProbClassDict[classId][wordId])
			# print(score)
			if score > maxScore:
				maxScore = score
				bestClass = classId

		predictionsDict[test_docId] = bestClass

	return predictionsDict
		
def accuracy(predictionsDict, Vsize):
	#count rights
	right = 0
	#get total counts
	total = len(docClassDict)

	# iterate over list and check if present then increase count
	for doc in docClassDict:

		if doc in predictionsDict:
			if predictionsDict[doc] == docClassDict[doc]:
				right += 1


	print("Accuracy = ", right/total)





"""
----------------------------RUN-----------------------
"""

# Vocab for train.data 
TrainVocab = createVocab()

# we sort the keys on basis of the values i.e frequencies.
TrainVocabSorted = sortVocab(TrainVocab)

# we get priorProbDict and classDict after prior probab claculations.
priorProbDict, classDict = priorProb()

# we get document and class dictionary for test.labels
docClassDict = createDocClassDict(Y_test)

# we get dictionary for test.data
test_dict = createTestDict()



# different sizes for vocabs that we have to iterate to control the size for vocab 
# that goes to Bernoulli distribution.
# Vsizes = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, len(TrainVocab)]
Vsizes = [100, 500]

for Vsize in Vsizes:

	#here we add all new words and set so that they are unique
	selectedVocab = set()

	for i in range(Vsize):
		#get Train vocabs values at [0] i.e the wordId and append it to train selectedVocab
		# this gives us a set that stroes all unique values
		selectedVocab.add(TrainVocabSorted[i][0])

	#this stores the list of classes as expalined in function desc.
	classList = BernoulliDist(selectedVocab)
	#this stores the list of conditional prob for each class.
	condProbClassDict = ConditionalProbDict(selectedVocab, classDict, classList)

	predictionsDict = predict(selectedVocab,Vsize, condProbClassDict, priorProbDict)
	accuracy(predictionsDict, Vsize)

