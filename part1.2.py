import numpy as np
from copy import copy, deepcopy
import time
from collections import defaultdict as setdefault
import itertools

# Overlapping Grids
alphabets = [' ', '+', '#']
keywords = itertools.product(alphabets, repeat = 9)
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

	with open('testimages') as f:
		content_testing = f.readlines()

	print(trainingprior)

	Building_group(content_training, traininglabels, NavieDic_group)
	Building_group_prior(content_training, traininglabels, NavieDic_prior)

	poss_laplace = [0.1, 0.2, 0.5, 1, 3, 10]
	for i in range(len(poss_laplace)):
		NavieClassify_group(content_testing, NavieDic_group, NavieDic_prior, testresult, trainingset, trainingprior, poss_laplace[i])
		correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])
		Accuracy_rate = (correct_predicted/len(testlabels))*100
		print("Accuracy (Laplace = ", poss_laplace[i], "): ", Accuracy_rate, "%")
		testresult = []

	# for i in range(len(testresult)):
	# 	if testresult[i] == testresult_group[i]:
	# 		continue
	# 	else:
	# 			testresult[i] = testresult_group[i]


	# correct_predicted = len([i for i, j in zip(traininglabels, testresult) if i == j])

	# error = np.zeros((10))
	# for i in range(len(traininglabels)):
	# 	if(traininglabels[i] != testresult[i]):
	# 		error[traininglabels[i]] += 1

	# Accuracy_rate = (correct_predicted/len(traininglabels))*100
	# end = time.time()
	# print("Accuracy Rate: ", Accuracy_rate, "%")
	# print(error)
	# print("Time: ", end - start_time)

	# correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])

	# error = np.zeros((10))
	# for i in range(len(testlabels)):
	# 	if(testlabels[i] != testresult[i]):
	# 		error[testlabels[i]] += 1

	# Accuracy_rate = (correct_predicted/len(testlabels))*100
	# end = time.time()
	# print("Accuracy Rate: ", Accuracy_rate, "%")
	# print(error)
	# print("Time: ", end - start_time)

def Building_group_prior(content, traininglabels, NavieDic_prior):
	class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	for i in range(int(len(content)/28)):
		cur_label = traininglabels[i]
		for row in range(i*28, (i+1)*28-2):
			for col in range(26):
				loc = (row%26, col) 			# location of that grid
				
				if loc not in NavieDic_prior:
					NavieDic_prior.setdefault(loc, {})
					for c in class_list:
						NavieDic_prior[loc][c] = 0
					NavieDic_prior[loc][cur_label] += 1
				else:
					NavieDic_prior[loc][cur_label] += 1
	# import IPython
	# IPython.embed()
	# exit()			
def Building_group(content, traininglabels, NavieDic):
	for i in range(int(len(content)/28)):
		cur_label = traininglabels[i]
		for row in range(i*28, (i+1)*28-2):
			for col in range(26):

				loc = (row%26, col) 			# location of that grid
				cur_tuple = (content[row][col], content[row][col+1], content[row][col+2], content[row+1][col], content[row+1][col+1], \
					content[row+1][col+2], content[row+2][col], content[row+2][col+1], content[row+2][col+2])
							# content[row+2][col], content[row+2][col+1], content[row+2][col+2], content[row+2][col+3], \
							# content[row+3][col], content[row+3][col+1], content[row+3][col+2], content[row+3][col+3])

				if loc not in NavieDic:			# not in the dictionary yet
					NavieDic.setdefault(loc, {})
					for c in range(len(comb_list)):
						NavieDic[loc][comb_list[c]] = np.zeros((10))

					NavieDic[loc][cur_tuple][cur_label] += 1
				else:
					NavieDic[loc][cur_tuple][cur_label] += 1

def NavieClassify_group(content, NavieDic, NavieDic_prior, testresult, trainingset, trainingprior, laplace_const):
	for i in range(int(len(content)/28)):

		posteriori = [np.log(trainingprior[i]) for i in range(10)]
		for row in range(i*28, (i+1)*28-2):
			for col in range(26):


				loc = (row%26, col) 			# location of that grid
				cur_tuple = (content[row][col], content[row][col+1], content[row][col+2], content[row+1][col], content[row+1][col+1], \
					content[row+1][col+2], content[row+2][col], content[row+2][col+1], content[row+2][col+2])
							# content[row+2][col], content[row+2][col+1], content[row+2][col+2], content[row+2][col+3], \
							# content[row+3][col], content[row+3][col+1], content[row+3][col+2], content[row+3][col+3])
		
				classlist = NavieDic[loc][cur_tuple]

				for i in range(10):
					total_num = NavieDic_prior[loc][i]
					lkhood = classlist[i]

					if lkhood == 0:
						lkhood = laplace_const
						total_num += laplace_const*2

					posteriori[i] += np.log(lkhood/total_num)

		max_label = posteriori.index(max(posteriori))
		testresult.extend([max_label])

def Building(content, traininglabels, NavieDic):
	for i in range(int(len(content)/28)):
		cur_label = traininglabels[i]
		for row in range(i*28, (i+1)*28):
			for col in range(28):
				loc = (row%28, col) 			# location of that grid
				cur_char = content[row][col]	# character in that grid
				if loc not in NavieDic:			# not in the dictionary yet
					NavieDic.setdefault(loc, {})
					NavieDic[loc][' '] = np.zeros((10))
					NavieDic[loc]['+'] = np.zeros((10))
					NavieDic[loc]['#'] = np.zeros((10))
					NavieDic[loc][cur_char][cur_label] += 1
				else:
					NavieDic[loc][cur_char][cur_label] += 1

def NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, laplace_const):
	for i in range(int(len(content)/28)):

		posteriori = [np.log(trainingprior[i]) for i in range(10)]
		cur_set = deepcopy(trainingset)
		for row in range(i*28, (i+1)*28):
			for col in range(28):
				loc = (row%28, col) 			# location of that grid
				cur_char = content[row][col]	# character in that grid
			
				classlist = NavieDic[loc][cur_char]

				for i in range(10):
					lkhood = classlist[i]
					if lkhood == 0:
						lkhood = laplace_const
						cur_set[i] += laplace_const
					posteriori[i] += np.log(lkhood/cur_set[i])
		max_label = posteriori.index(max(posteriori))
		testresult.extend([max_label])

# def NavieClassify_joint(content, NavieDic, NavieDic_group, testresult, trainingset, trainingprior, laplace_const_ind, laplace_const_group):
# 	for i in range(int(len(content)/28)):

# 		posteriori = [np.log(trainingprior[i]) for i in range(10)]
# 		cur_set = deepcopy(trainingset)
		
# 		for row in range(i*28, (i+1)*28-2):
# 			for col in range(26):


# 				loc = (row%26, col) 			# location of that grid

# 				cur_char = content[row][col]
# 				cur_tuple = (content[row][col], content[row][col+1], content[row][col+2], \
# 							 content[row+1][col], content[row+1][col+1], content[row+1][col+2] \
# 							 , content[row+2][col], content[row+2][col+1], content[row+2][col+2])		

# 				classlist_ind = NavieDic[loc][cur_char]
# 				classlist_group = NavieDic_group[loc][cur_tuple]

# 				for i in range(10):
# 					lkhood_ind = classlist_ind[i]
# 					lkhood_group = classlist_group[i]

# 					total_num_ind = cur_set[i]
# 					total_num_group = cur_set[i]

# 					if lkhood_ind == 0:
# 						lkhood_ind = laplace_const_ind
# 						total_num_ind += laplace_const_ind

# 					if lkhood_group == 0:
# 						lkhood_group = laplace_const_group
# 						total_num_group += laplace_const_group

# 					posteriori[i] += np.log((lkhood_ind )/total_num_ind)

# 					# posteriori[i] += np.log(2.42) * np.log(lkhood_ind/total_num_ind) + np.log(6) *  np.log(lkhood_group/total_num_group)

# 		max_label = posteriori.index(max(posteriori))
# 		testresult.extend([max_label])

main()


