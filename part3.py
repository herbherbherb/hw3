import numpy as np
from copy import copy, deepcopy
import time, sys
from collections import defaultdict as setdefault
import itertools
# import ipdb
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
#=======================Using Multinomial Navie Bayes===========================================
	if arg == "1":
		print("Multinomial Navie Bayes")
		
		# best_accuracy = 0
		# best_laplace = 0
		# c = [0.5, 1, 2, 5, 10, 100, 1000, 5000]
		# for i in c:
		# 	pclass = {}
		# 	nclass = {}
		# 	accuracy = 0
		# 	Build_Mult(content_movie, pclass, nclass, i)
		# 	accuracy = Classify_Mult(content_movie_testing, pclass, nclass, movie_p_prior, movie_n_prior)
		# 	# import IPython
		# 	# IPython.embed()
		# 	# exit()
		# 	print("Accuracy ", i, ": ", accuracy, "%")
		# 	if best_accuracy < accuracy:
		# 		best_accuracy = accuracy
		# 		best_laplace = i
		# print("Best Laplace: ", best_laplace)
		# print("Movie Reviews Accuracy: ", best_accuracy, "%")


		best_accuracy = 0
		best_laplace = 0
		c = [0.5, 1, 2, 5, 10, 100, 1000, 5000]
		for i in c:
			pclass = {}
			nclass = {}
			accuracy = 0
			Build_Mult(content_topic, pclass, nclass, i)
			accuracy = Classify_Mult(content_topic_testing, pclass, nclass, topic_p_prior, topic_n_prior)
			print("Accuracy ", i, ": ", accuracy, "%")
			if best_accuracy < accuracy:
				best_accuracy = accuracy
				best_laplace = i
		print("Best Laplace: ", best_laplace)
		print("Binary Conversation Accuracy: ", best_accuracy, "%")

#=======================Using Bernoulli Navie Bayes==============================================
	if arg == "2":
		print("Bernoulli Navie Bayes")
		best_accuracy = 0
		best_laplace = 0
		c = [1, 2, 5, 10, 100, 1000, 9075]
		for i in c:
			pclass = {}
			nclass = {}
			accuracy = 0
			word_bank = []
			p_word_bank = []
			n_word_bank = []
			accuracy = 0
			Build_Bernoulli(content_topic, pclass, nclass, word_bank, p_word_bank, n_word_bank, i)
			accuracy = accuracy = Classify_Bernoulli(content_topic_testing, pclass, nclass, movie_p_prior, movie_n_prior, p_word_bank, n_word_bank, word_bank)
		
			print("Accuracy ", i, ": ", accuracy, "%")
			if best_accuracy < accuracy:
				best_accuracy = accuracy
				best_laplace = i
		print("Best Laplace: ", best_laplace)
		print("Movie Reviews Accuracy: ", best_accuracy, "%")


		# pclass = {}
		# nclass = {}
		# word_bank = []
		# p_word_bank = []
		# n_word_bank = []
		# Build_Bernoulli(content_topic, pclass, nclass, word_bank, p_word_bank, n_word_bank, 1)
		# accuracy = Classify_Bernoulli(content_topic_testing, pclass, nclass, movie_p_prior, movie_n_prior, p_word_bank, n_word_bank, word_bank)
		# print("Binary Conversation Accuracy: ", accuracy, "%")

def Classify_Mult(content_testing, pclass, nclass, p_prior, n_prior):
	accuracy = 0
	for line in content_testing:
		# pclass_value = np.log(p_prior)
		# nclass_value = np.log(n_prior)
		pclass_value = 0
		nclass_value = 0

		testlabels = -1
		if line[0] == "1":
			testlabels = 1
			line = line[2:]
		else:
			line = line[3:]

		for i in line.split(' '):
			word = i.split(':')[0]
			count = i.split(':')[1]

			if word in pclass:
				pclass_value += (pclass[word]) * int(count)
			if word in nclass:
				nclass_value += (nclass[word]) * int(count)
		
		if pclass_value > nclass_value and testlabels == 1:
			accuracy += 1
	
		if nclass_value > pclass_value and testlabels == -1:
			accuracy += 1

	accuracy /= len(content_testing)
	return (accuracy * 100)

def Build_Mult(content, pclass, nclass, laplace):

	pclass_total = 0
	nclass_total = 0
	total_unique = []
	
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

def Classify_Bernoulli(content_testing, pclass, nclass, p_prior, n_prior, p_word_bank, n_word_bank, word_bank):
	accuracy = 0
	remaining = deepcopy(word_bank)
	import IPython
	IPython.embed()
	exit()
	
	for line in content_testing:
		pclass_value = np.abs(np.log(p_prior))
		nclass_value = np.abs(np.log(n_prior))

		testlabels = -1
		if line[0] == "1":
			testlabels = 1
			line = line[2:]
		else:	
			line = line[3:]

		for i in line.split(' '):
			word = i.split(':')[0]
			count = i.split(':')[1]
			if word not in word_bank:
				continue
			remaining.remove(word)
			if word in pclass:
				pclass_value += pclass[word]
			if word in nclass:
				nclass_value += nclass[word]

		for remain in remaining:
			pclass_value += pclass[remain]
			nclass_value += nclass[remain]

		if pclass_value > nclass_value and testlabels == 1:
			accuracy += 1

		if nclass_value > pclass_value and testlabels == -1:
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
		if word in p_word_bank:
			count = pclass[word]
			count += laplace
			count /= (440 + laplace * 2)
			pclass[word] = np.log10(count)
		else:
			count = laplace/(440 + laplace * 2)
			pclass[word] = np.log10(count)

		if word in n_word_bank:
			count = pclass[word]
			count += laplace
			count /= (438 + laplace * 2)
			nclass[word] = np.log10(count)
		else:
			count = laplace/(438 + laplace * 2)
			nclass[word] = np.log10(count)

main()
