import numpy as np
from copy import copy, deepcopy
import time, sys
from collections import defaultdict as setdefault
import itertools, operator
import pdb
def main():
	with open('rt-train.txt') as f:
			content_movie = f.readlines()
	with open('rt-test.txt') as f:
			content_movie_testing = f.readlines()

	with open('fisher_train_2topic.txt') as f:
			content_topic = f.readlines()
	with open('fisher_test_2topic.txt') as f:
			content_topic_testing = f.readlines()

	movie_p_prior = movie_n_prior = 0.5
	topic_p_prior = (440 / 878)
	topic_n_prior = (438 / 878)	
	# print(topic_p_prior)
	# print(topic_n_prior)

	if len(sys.argv) > 1:
		arg = sys.argv[1]
	else:
		arg = 0
#=======================Using Multinomial Naive Bayes===========================================
	if arg == "1":
		print("Multinomial Naive Bayes")
		
		# best_accuracy = 0
		# best_laplace = 0
		# c = [2]
		# for i in c:
		# 	testlabels = []
		# 	testresult = []
		# 	total_unique = []
		# 	pclass = {}
		# 	nclass = {}
		# 	accuracy = 0
		# 	Build_Mult(content_movie, pclass, nclass, i, total_unique)
		# 	accuracy = Classify_Mult(content_movie_testing, pclass, nclass, \
		# 					movie_p_prior, movie_n_prior, testlabels, testresult)
			
		# 	confusion_mtx(testlabels, testresult, pclass, nclass, total_unique)

		# 	print("Accuracy ", i, ": ", accuracy, "%")
		# 	# if best_accuracy < accuracy:
		# 	# 	best_accuracy = accuracy
		# 	# 	best_laplace = i
		# # print("Best Laplace Value: ", best_laplace)
		# # print("Movie Reviews Accuracy: ", best_accuracy, "%")


		best_accuracy = 0
		best_laplace = 0
		c = [100]
		for i in c:
			testlabels = []
			testresult = []
			total_unique = []
			pclass = {}
			nclass = {}
			accuracy = 0
			Build_Mult(content_topic, pclass, nclass, i, total_unique)
			accuracy = Classify_Mult(content_topic_testing, pclass, nclass, \
							topic_p_prior, topic_n_prior, testlabels, testresult)

			confusion_mtx(testlabels, testresult, pclass, nclass, total_unique)

			print("Accuracy ", i, ": ", accuracy, "%")
		# 	if best_accuracy < accuracy:
		# 		best_accuracy = accuracy
		# 		best_laplace = i
		# print("Best Laplace Value: ", best_laplace)
		# print("Binary Conversation Accuracy: ", best_accuracy, "%")

#=======================Using Bernoulli Naive Bayes==============================================
	if arg == "2":
		print("Bernoulli Naive Bayes")
		
		# best_accuracy = 0
		# best_laplace = 0
		# c = [3]
		# for i in c:
		# 	pclass = {}
		# 	nclass = {}
		# 	accuracy = 0
		# 	word_bank = []
		# 	p_word_bank = []
		# 	n_word_bank = []
		# 	testlabels = []
		# 	testresult = []
		# 	accuracy = 0
		# 	Build_Bernoulli(content_movie, pclass, nclass, word_bank, p_word_bank, n_word_bank, i)
		# 	accuracy = Classify_Bernoulli(content_movie_testing, pclass, nclass, \
		# 				movie_p_prior, movie_n_prior, p_word_bank, n_word_bank, word_bank, i, testlabels, testresult)
			
		# 	confusion_mtx(testlabels, testresult, pclass, nclass, word_bank)
		# 	print("Accuracy ", i, ": ", accuracy, "%")
		# # 	if best_accuracy < accuracy:
		# # 		best_accuracy = accuracy
		# # 		best_laplace = i
		# # print("Best Laplace Value: ", best_laplace)
		# # print("Movie Reviews Accuracy: ", best_accuracy, "%")


		best_accuracy = 0
		best_laplace = 0
		c = [0.1]
		for i in c:
			pclass = {}
			nclass = {}
			word_bank = []
			p_word_bank = []
			n_word_bank = []
			testlabels = []
			testresult = []
			Build_Bernoulli(content_topic, pclass, nclass, word_bank, p_word_bank, n_word_bank, i)
			accuracy = Classify_Bernoulli(content_topic_testing, pclass, nclass, \
								movie_p_prior, movie_n_prior, p_word_bank, n_word_bank, word_bank, i, testlabels, testresult)
			

			confusion_mtx(testlabels, testresult, pclass, nclass, word_bank)
			print("Accuracy ", i, ": ", accuracy, "%")
		# 	if best_accuracy < accuracy:
		# 		best_accuracy = accuracy
		# 		best_laplace = i
		# print("Best Laplace: ", best_laplace)
		# print("Binary Conversation Accuracy: ", best_accuracy, "%")

def confusion_mtx(testlabels, testresult, pclass, nclass, word_bank):
#====================Confusion Matrix========================================
	confusion = np.zeros((3, 3))
	confusion[0, 1:] = [1, -1]
	confusion[1:, 0] = [1, -1]
	for i in range(len(testlabels)):
		if testlabels[i] == 1:
			if testresult[i] == 1:
				confusion[1, 1] += 1
			else:
				confusion[1, 2] += 1
		else:
			if testresult[i] == 1:
				confusion[2, 1] += 1
			else:
				confusion[2, 2] += 1

	confusion[1:, 1:] /= len(testresult)
	np.set_printoptions(precision=3)
	print("Confusion Matrix:")
	print(confusion)
	print()

#=====================Top 10 Liklihood Words===================================
	sorted_p = sorted(pclass.items(), key=operator.itemgetter(1), reverse = True)
	sorted_n = sorted(nclass.items(), key=operator.itemgetter(1), reverse = True)

	print("Top 10 Words with Highest Likelihood in Positive Class :")
	for i in range(10):
		np.set_printoptions(precision=3)
		print(sorted_p[i], end = " ")
		if i == 1 or i == 3 or i == 5 or i == 7 or i == 9:
			print()
	print()
	print("Top 10 Words with Highest Likelihood in Negative Class  :")
	for i in range(10):
		np.set_printoptions(precision=3)
		print(sorted_n[i], end = " ")
		if i == 1 or i == 3 or i == 5 or i == 7 or i == 9:
			print()
	print()
#=========================Top 10 Odd Ratio======================================
	print("Top 10 Odd Ratio:")
	odd_ratio = []
	for i in range(len(word_bank)):	
		cur_word = word_bank[i]
		if nclass[cur_word] != 0:
			ratio = pclass[cur_word] / nclass[cur_word]
			odd_ratio.extend([(cur_word, ratio)])
		# else:
		# 	ratio = pclass[cur_word] / (nclass[cur_word]+1)
		# 	odd_ratio.extend([(cur_word, ratio)])

	odd_ratio = sorted(odd_ratio, key=lambda x: x[1], reverse= True)

	for i in range(10):
		np.set_printoptions(precision=3)
		print(odd_ratio[i], end = " ")
		if i == 1 or i == 3 or i == 5 or i == 7 or i == 9:
			print()
	print()


def Classify_Mult(content_testing, pclass, nclass, p_prior, n_prior, testlabels, testresult):
	accuracy = 0
	for line in content_testing:
		# pclass_value = np.log(p_prior)
		# nclass_value = np.log(n_prior)
		pclass_value = 0
		nclass_value = 0

		currlabels = -1
		if line[0] == "1":
			currlabels = 1
			line = line[2:]
		else:
			line = line[3:]

		testlabels.extend([currlabels])

		for i in line.split(' '):
			word = i.split(':')[0]
			count = i.split(':')[1]

			if word in pclass:
				pclass_value += (pclass[word]) * int(count)
			if word in nclass:
				nclass_value += (nclass[word]) * int(count)
		
		if pclass_value >= nclass_value:
			testresult.extend([1])
		else:
			testresult.extend([-1])

		if pclass_value >= nclass_value and currlabels == 1:
			accuracy += 1
		if nclass_value >= pclass_value and currlabels == -1:
			accuracy += 1

	accuracy /= len(content_testing)
	return (accuracy * 100)

def Build_Mult(content, pclass, nclass, laplace, total_unique):

	pclass_total = 0
	nclass_total = 0

	for line in content:
		if line[0] == "1":
			line = line[2:]
			for i in line.split(' '):
				word = i.split(':')[0]
				count = i.split(':')[1]

				pclass_total += int(count)
				if word not in pclass and count != 0:
					pclass[word] = int(count)
				else:
					pclass[word] += int(count)
				if word not in set(total_unique):
					total_unique.extend([word])
		else:
			line = line[3:]
			for i in line.split(' '):
				word = i.split(':')[0]	
				count = i.split(':')[1]

				nclass_total += int(count)
				if word not in nclass and count != 0:
					nclass[word] = int(count)
				else:
					nclass[word] += int(count)

				if word not in set(total_unique):
					total_unique.extend([word])

	for word in total_unique:
		if word in pclass:
			count = pclass[word]
			count += laplace
			count /= (pclass_total + laplace * len(total_unique))
			pclass[word] = np.log10(count)
		else:
			count = laplace/(pclass_total + laplace * len(total_unique))
			pclass[word] = np.log10(count)

		if word in nclass:
			count = nclass[word]
			count += laplace
			count /= (nclass_total + laplace * len(total_unique))
			nclass[word] = np.log10(count)
		else:
			count = laplace/(nclass_total + laplace * len(total_unique))
			nclass[word] = np.log10(count)

#===========================================================================================================================
def Classify_Bernoulli(content_testing, pclass, nclass, p_prior, n_prior, p_word_bank, n_word_bank, word_bank, laplace, testlabels, testresult):
	accuracy = 0
	for line in content_testing:
		remaining = deepcopy(word_bank)
		pclass_value = np.log(p_prior)
		nclass_value = np.log(n_prior)

		currlabels = -1
		if line[0] == "1":
			currlabels = 1
			line = line[2:]
		else:	
			line = line[3:]

		testlabels.extend([currlabels])

		for i in line.split(' '):
			word = i.split(':')[0]
			if word not in word_bank:
				continue
			if word in remaining:
				remaining.remove(word)

			if word in pclass:
				count = pclass[word]
				count += laplace
				count /= (440 + laplace * 2)
				pclass_value += np.log(count)   #log
			if word in nclass:
				count = nclass[word]
				count += laplace
				count /= (438 + laplace * 2)    
				nclass_value += np.log(count)	#log

		for remain in remaining:
			count = (pclass[remain] + laplace)/(440 + laplace * 2)
			pclass_value += np.log10(1-count)		#log10
			count = (nclass[remain] + laplace)/(438 + laplace * 2)
			nclass_value += np.log10(1-count)		#log10

		if pclass_value >= nclass_value:
			testresult.extend([1])
		else:
			testresult.extend([-1])

		if pclass_value >= nclass_value and currlabels == 1:
			accuracy += 1

		if nclass_value >= pclass_value and currlabels == -1:
			accuracy += 1

	accuracy /= len(content_testing)
	return accuracy * 100

def Build_Bernoulli(content, pclass, nclass, word_bank, p_word_bank, n_word_bank, laplace):
	for line in content:
		if line[0] == "1":
			line = line[2:]
			for i in line.split(' '):
				word = i.split(':')[0]
				count = i.split(':')[1]
				if word not in pclass and count != 0:
					pclass[word] = 1
				else:
					pclass[word] += 1

				if word not in set(word_bank):
					word_bank.extend([word])
		else:
			line = line[3:]
			for i in line.split(' '):
				word = i.split(':')[0]	
				count = i.split(':')[1]
				if word not in nclass and count != 0:
					nclass[word] = 1
				else:
					nclass[word] += 1

				if word not in set(word_bank):
					word_bank.extend([word])

	for word in word_bank:
		if word not in pclass:
			pclass[word] = 0
		if word not in nclass:
			nclass[word] = 0
main()
