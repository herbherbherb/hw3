import numpy as np
from copy import copy, deepcopy
import time
from collections import defaultdict as setdefault
import itertools

# Overlapping Grid+s
alphabets = [' ', '#']
keywords = itertools.product(alphabets, repeat = 4)
comb_list = list(keywords)

def main():
#=======initialize================
	start_time = time.time()
	traininglabels = []
	testlabels = []
	trainingprior = np.zeros((10))
	likelihood = []
	testresult = []
	testresult_group = []

	NavieDic = {}
	NavieDic_prior = {}
	NavieDic_group = {}
	
#========Reading==================
	with open('traininglabels') as f:
		content = f.readlines()
	traininglabels = [int(x.strip('\n')) for x in content]

	with open('testlabels') as f:
		content = f.readlines()
	testlabels = [int(x.strip('\n')) for x in content]


	trainingset = {}						# P(piror) = trainingprior[i] # of class[i] = trainingset
	traininglength = len(traininglabels)
	for i in range(10):
		count = traininglabels.count(i)
		trainingset[i] = count
		trainingprior[i] = count/traininglength
		likelihood.append([])

	with open('trainingimages') as f:
		content_training = f.readlines()

	for i in range(len(content_training)):
		content_training[i] = ['#' if x=='+' else x for x in content_training[i]]

	with open('testimages') as f:
		content_testing = f.readlines()

	for i in range(len(content_testing)):
		content_testing[i] = ['#' if x=='+' else x for x in content_testing[i]]

	Label_count = np.zeros((10))
	Building_group(content_training, traininglabels, NavieDic_group, Label_count)
	# import IPython
	# IPython.embed()
	# exit()
	# Building_group_prior(content_training, traininglabels, NavieDic_prior, NavieDic_group)

	poss_laplace = [0.1, 0.2, 0.5, 1, 3, 10]
	for i in range(len(poss_laplace)):
		NavieClassify_group(content_testing, NavieDic_group, Label_count, testresult, trainingset, trainingprior, poss_laplace[i])
		correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])
		Accuracy_rate = (correct_predicted/len(testlabels))*100
		print("Accuracy (Laplace = ", poss_laplace[i], "): ", Accuracy_rate, "%")
		testresult = []

# def Building_group_prior(content, traininglabels, dic, NavieDic):



	# for i in range(27):
	# 	for j in range(27):
	# 		loc = (i, j)
	# 		dic.setdefault(loc, {})
	# 		for label in range(10):
	# 			dic[loc][label] = 0
	# 			count = 0
	# 			for char in NavieDic[loc]:
	# 				count += NavieDic[loc][char][label]
	# 			dic[loc][label] = count

def Building_group(content, traininglabels, NavieDic, Label_count):
	for i in range(int(len(content)/28)):
		cur_label = traininglabels[i]
		Label_count[cur_label] += 1
		for row in range(i*28, (i+1)*28-1):
			for col in range(27):

				loc = (row%27, col) 			# location of that grid
				cur_tuple = (content[row][col], content[row][col+1], content[row+1][col], content[row+1][col+1])
					# content[row+1][col+2], content[row+2][col], content[row+2][col+1], content[row+2][col+2])
							# content[row+2][col], content[row+2][col+1], content[row+2][col+2], content[row+2][col+3], \
							# content[row+3][col], content[row+3][col+1], content[row+3][col+2], content[row+3][col+3])

				if loc not in NavieDic:			# not in the dictionary yet
					NavieDic.setdefault(loc, {})
					for c in range(len(comb_list)):
						NavieDic[loc][comb_list[c]] = np.zeros((10))

					NavieDic[loc][cur_tuple][cur_label] += 1
				else:
					NavieDic[loc][cur_tuple][cur_label] += 1

def NavieClassify_group(content, NavieDic, Label_count, testresult, trainingset, trainingprior, laplace_const):
	for i in range(int(len(content)/28)):

		posteriori = [np.log(trainingprior[i]) for i in range(10)]
		for row in range(i*28, (i+1)*28-1):
			for col in range(27):


				loc = (row%27, col) 			# location of that grid
				cur_tuple = (content[row][col], content[row][col+1], content[row+1][col], content[row+1][col+1])
					# content[row+1][col+2], content[row+2][col], content[row+2][col+1], content[row+2][col+2])
							# content[row+2][col], content[row+2][col+1], content[row+2][col+2], content[row+2][col+3], \
							# content[row+3][col], content[row+3][col+1], content[row+3][col+2], content[row+3][col+3])
		
				classlist = NavieDic[loc][cur_tuple]

				for i in range(10):
					total_num = Label_count[i]
					lkhood = classlist[i]

					lkhood += laplace_const
					total_num += laplace_const*(2**4)

					posteriori[i] += np.log(lkhood/total_num)

		max_label = posteriori.index(max(posteriori))
		testresult.extend([max_label])

main()


