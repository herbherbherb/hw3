import numpy as np
from copy import copy, deepcopy
import time
from collections import defaultdict as setdefault
import itertools


def main():
	accuracy = 0
	with open('rt-train.txt') as f:
		content = f.readlines()
	with open('rt-test.txt') as f:
		content_testing = f.readlines()


	adjust_value = 0
	best_laplace = 0
	best_accuracy = 0
	for i in range(1, 5, 1):
		pclass = {}
		nclass = {}
		pclass_total = 0
		nclass_total = 0		
		pclass_unique = 0
		nclass_unique = 0
		BuildDic(content, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique, i, 23500)
		accuracy = Classify(content_testing, accuracy, pclass, nclass, pclass_total, nclass_total, pclass_unique, nclass_unique)
		if best_accuracy < accuracy:
			best_accuracy = accuracy
			best_laplace = i
	print("Laplace Value: ", best_laplace)
	print("The Accuracy is: ", best_accuracy, "%")

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
				pclass_value *= (pclass[word]**int(count))
			if word in nclass:
				nclass_value *= (nclass[word]**int(count))

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
		count /= (pclass_unique + pclass_total)
		pclass[k] = count

	for k in nclass:
		count = nclass[k]
		count = (count + laplace_const) * adjust
		count /= (nclass_unique + nclass_total)
		nclass[k] = count

main()
