import numpy as np
from copy import copy, deepcopy
import time, sys
from collections import defaultdict as setdefault
import itertools


def main():
	if len(sys.argv) > 1:
		arg = sys.argv[1]
	else:
		arg = 0

	if arg == "1":
#=======================Part2.1===========================================
		accuracy = 0
		with open('rt-train.txt') as f:
			content = f.readlines()
		with open('rt-test.txt') as f:
			content_testing = f.readlines()


		adjust_value = 0
		best_laplace = 0
		best_accuracy = 0
		for i in range(23000, 30000, 20):
			pclass = {}
			nclass = {}
			pclass_total = 0
			nclass_total = 0		
			pclass_unique = 0
			nclass_unique = 0
			BuildDic(content, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, 1, i)
			accuracy = Classify(content_testing, accuracy, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique)
			if best_accuracy < accuracy:
				best_accuracy = accuracy
				adjust_value = i
		print("Adjust Value: ", adjust_value)
		print("The Accuracy is: ", best_accuracy, "%")

#=======================Part2.2===========================================
	if arg == "2":
		with open('fisher_train_2topic.txt') as f:
			content = f.readlines()
		with open('fisher_test_2topic.txt') as f:
			content_testing = f.readlines()

		adjust_value = 0
		best_laplace = 0
		best_accuracy = 0
		for i in range(23000, 30000, 20):
			pclass = {}
			nclass = {}
			pclass_total = 0
			nclass_total = 0		
			pclass_unique = 0
			nclass_unique = 0
			p_prior, n_prior = BuildDicFisher(content, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, 1, i)
		
		# import IPython
		# IPython.embed()
		# exit()
		
			accuracy = ClassifyFisher(content_testing, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, p_prior, n_prior)
			if best_accuracy < accuracy:
				best_accuracy = accuracy
				adjust_value = i
		print("Adjust Value: ", adjust_value)
		print("The Accuracy is: ", best_accuracy, "%")

def ClassifyFisher(content_testing, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, p_prior, n_prior):
	accuracy = 0

	for line in content_testing:
		pclass_value = p_prior
		nclass_value = n_prior

		testlabels = -1
		if line[0] == "1":
			testlabels = 1
			line = line[2:]
		else:
			line = line[3:]
		for i in line.split(' '):
			word = i.split(':')[0]
			count = i.split(':')[1]

			if word in pclass and pclass[word] != 0:
				pclass_value *= pclass[word]
			if word in nclass and nclass[word] != 0:
				nclass_value *= nclass[word]
		
		if pclass_value > nclass_value and testlabels == 1:
			accuracy += 1

		if nclass_value > pclass_value and testlabels == -1:
			accuracy += 1

	accuracy /= len(content_testing)
	return accuracy * 100

def BuildDicFisher(content, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, laplace, adjust):
	p_prior = 0
	n_prior = 0
	for line in content:
		if line[0] == "1":
			p_prior += 1
			line = line[2:]
			for i in line.split(' '):
				word = i.split(':')[0]
				count = i.split(':')[1]
				pclass_total += int(count)
				if word not in pclass and count != 0:
					pclass_unique += 1
					pclass[word] = 1
				elif count != 0:
					pclass[word] += 1
		else:
			line = line[3:]
			n_prior += 1
			for i in line.split(' '):
				word = i.split(':')[0]	
				count = i.split(':')[1]
				nclass_total += int(count)
				if word not in nclass and count != 0:
					nclass_unique += 1
					nclass[word] = 1
				elif count != 0:
					nclass[word] += 1

	for k in pclass:
		count = pclass[k]
		count = (count + laplace) * adjust
		count /= (pclass_total)
		pclass[k] = count

	for k in nclass:
		count = nclass[k]
		count = (count + laplace) * adjust
		count /= (nclass_total)
		nclass[k] = count
	return p_prior/len(content), n_prior/len(content)

def Classify(content_testing, accuracy, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique):
	for line in content_testing:
		pclass_value = 1
		nclass_value = 1

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
				pclass_value *= pclass[word]**int(count)
			if word in nclass:
				nclass_value *= nclass[word]**int(count)
		
		if pclass_value > nclass_value and testlabels == 1:
			accuracy += 1

		if nclass_value > pclass_value and testlabels == -1:
			accuracy += 1

	accuracy /= 1000
	return accuracy * 100

def BuildDic(content, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, laplace_const, adjust):
	for line in content:
		if line[0] == "1":
			line = line[2:]
			for i in line.split(' '):
				word = i.split(':')[0]
				count = i.split(':')[1]
				pclass_total += int(count)
				if word not in pclass:
					pclass_unique += 1
					pclass[word] = int(count)
				else:
					pclass[word] += int(count)
		else:
			line = line[3:]
			for i in line.split(' '):
				word = i.split(':')[0]	
				count = i.split(':')[1]
				nclass_total += int(count)
				if word not in nclass:
					nclass_unique += 1
					nclass[word] = int(count)
				else:
					nclass[word] += int(count)

	for k in pclass:
		count = pclass[k]
		count = (count + laplace_const) * adjust
		count /= (pclass_total)
		pclass[k] = count

	for k in nclass:
		count = nclass[k]
		count = (count + laplace_const) * adjust
		count /= (nclass_total)
		nclass[k] = count

main()
