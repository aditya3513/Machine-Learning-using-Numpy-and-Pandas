import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import metrics
import matplotlib.pyplot as plt

#reading data in nd-Arrays
#reading data in nd-Arrays
data = np.genfromtxt ('../../wine.csv', delimiter=",")

#shuffle data
np.random.shuffle(data)

# nd array of Features
X = data[:, 1:].astype(np.float)

#nd array of Labels
Y = data[:, 0].flatten()

#for class 2 and all else nagative
for i in range(len(Y)):
	if Y[i] == 3:
		Y[i] = 1
	else:
		Y[i] = 0


def getGamma():
	gamma = []
	for i in np.arange(-5, 10, dtype=float):
		val = np.power(2,i)
		gamma.append(val)
	return gamma

def getC():
	C = []
	for i in np.arange(-15, 5, dtype=float):
		val = np.power(2,i)
		C.append(val)
	return C

def getCombinations(gamma, C):
	combinations = []

	for g in gamma:
		
		for c in C:
			combination = []
			combination.append(g)
			combination.append(c)
			
			combinations.append(combination)

	return combinations

def run():

	gamma = getGamma()
	C = getC()

	combinations = getCombinations(gamma, C) #[gamma, C] array

	final_acc = []
	final_recall = []
	final_precision = []


	#generating K folds, i.e 10 folds
	kf_k = KFold(n_splits=10)
	k = 1

	#this splis data in to 90:10 train-test split
	for train_index, test_index in kf_k.split(X):


		kf_m = KFold(n_splits=5)

		X_train, X_test = X[train_index], X[test_index]
		# Y_train = Y[train_index].flatten()
		# Y_test = Y[test_index].flatten()

		Y_train, Y_test = Y[train_index], Y[test_index]

		best_score = -999
		best_score_combo = [0, 0]

		for combination in combinations:

			scores = []

			for train_index_m, test_index_m in kf_m.split(X_train):

				X_train_m, X_test_m = X[train_index_m], X[test_index_m]
				Y_train_m = Y[train_index_m].flatten()
				Y_test_m = Y[test_index_m].flatten()


				scaler_m = StandardScaler().fit(X_train_m)
				X_train_m_scaled = scaler_m.transform(X_train_m)
				X_test_m_scaled = scaler_m.transform(X_test_m)

				X_train_m_scaled = np.insert(X_train_m_scaled, 0, np.ones(X_train_m.shape[0]), axis=1)

				X_test_m_scaled = np.insert(X_test_m_scaled, 0, np.ones(X_test_m.shape[0]), axis=1)

				model = SVC(gamma=combination[0], C=combination[1], kernel='rbf', max_iter=1000, probability=True)

				model.fit(X_train_m_scaled, Y_train_m)

				# score = model.score(X_test_m_scaled, Y_test_m)
				train_preds = model.predict_proba(X_test_m_scaled)[:,1]

				training_acc = accuracy_score(Y_test_m, train_preds.round())
				# train_fpr, train_tpr, train_threshold = metrics.roc_curve(Y_test_m, train_preds.round())
				
				# train_roc_auc = metrics.auc(train_fpr, train_tpr)

				scores.append(training_acc)

			mean_score = np.mean(scores)
			# print("----------Tuning [gamma, C] = ",combination, " score = ", mean_score)
			# print("mean score = ",mean_score)
			# combo_scores.append(mean_score)

			if mean_score > best_score:

				best_score = mean_score
				best_score_combo = combination

		print("------------------------------------------")
		print("Training results for fold = ",k, " [gamma, C] = ", best_score_combo)
		print("Final Results for all k folds")
		print("AUC score = ", best_score)


		#normalizing Training data
		scaler = StandardScaler().fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		X_train_scaled = np.insert(X_train_scaled, 0, np.ones(X_train.shape[0]), axis=1)

		X_test_scaled = np.insert(X_test_scaled, 0, np.ones(X_test.shape[0]), axis=1)
		
		Y_train = Y[train_index].flatten()
		Y_test = Y[test_index].flatten()

		model = SVC(gamma=best_score_combo[0], C=best_score_combo[1], kernel='rbf', max_iter=1000, probability=True)

		model.fit(X_train_scaled, Y_train)

		preds = model.predict_proba(X_test_scaled)[:,1]
		preds = preds.round()

		# preds = pred

		# actual_score = model.score(X_test_scaled, Y_test)
		# final_acc.append(actual_score)

		'''
		Now we calculate accuracy, recall, percision
		'''
		# accuracy
		validation_acc = accuracy_score(Y_test, preds)
		final_acc.append(validation_acc)

		# recall
		validation_recall = recall_score(Y_test, preds)
		final_recall.append(validation_recall)

		# precision
		validation_precision = precision_score(Y_test, preds)
		final_precision.append(validation_precision)


		print("\n\n Validation results for fold = ",k, " [gamma, C] = ", best_score_combo)
		print("Final Results for all k folds")
		print("accuracy = ", validation_acc)
		print("recall = ", validation_recall)
		print("precision = ", validation_precision)
		# print("Validation at fold k = ",k," best combination [gamma, C] = ",best_score_combo, " score = ", actual_score)
		'''
		Now we calculate auc, roc and curves
		'''
		
		fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
		df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
		roc_auc = metrics.auc(fpr, tpr)
		plt.figure()
		plt.title('Receiver Operating Characteristic')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		file_name = "fold_k_"+ str(k)+".png"
		plt.savefig(file_name)


		k = k + 1

	print("\n\n\n\n================================")
	print("Final Results for all k folds")
	print("mean accuracy = ", np.mean(final_acc))
	print("std dev. accuracy = ", np.std(final_acc))
	print("mean recall = ", np.mean(final_recall))
	print("std dev. recall = ", np.std(final_recall))
	print("mean precision = ", np.mean(final_precision))
	print("std dev. precision = ", np.std(final_precision))


	# print("Mean = ", np.mean(acc)*100)
	# print("Stand Deviation = ", np.std(acc)*100)

run()



