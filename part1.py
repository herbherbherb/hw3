import numpy as np
from copy import copy, deepcopy
import time
from collections import defaultdict as setdefault
# import matplotlib.pyplot as plt

def main():
#=======initialize================
	start_time = time.time()

	traininglabels = []
	testlabels = []
	trainingprior = np.zeros((10))
	likelihood = []
	testresult = []

	NavieDic = {}
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
		content = f.readlines()

	Building(content, traininglabels, NavieDic)
	end = time.time()
	print("Training Time: ", end - start_time)
#===========Testing================================================
	with open('testimages') as f:
		content = f.readlines()
	# poss_laplace = [0.1, 0.5, 1, 3, 5]
	poss_laplace = [0.5]
	for i in range(len(poss_laplace)):
		pos_low = np.zeros(10)
		pos_high = np.zeros(10)

		start_time = time.time()
		pos_high, pos_low = NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, \
					poss_laplace[i], pos_low, pos_high, testlabels)
		
		end = time.time()
		highlow(content, pos_high, pos_low)
		correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])

		Accuracy_rate = (correct_predicted/len(testlabels))*100
		print("Total Accuracy (Laplace = ", poss_laplace[i], "): ", Accuracy_rate, "%")
		report_digit_accuracy(testresult, testlabels, NavieDic, trainingset)
		print("Testing Time: ", end - start_time)
		testresult = []

def highlow(content, pos_high, pos_low):
	f = open('output.txt','w')
	for i in range(10):
		low = int(pos_low[i])
		high = int(pos_high[i])

		for c in range(low*28, (low+1)*28):
			f.write(content[c])
		for c in range(high*28, (high+1)*28):
			f.write(content[c])
	f.close()
	
def report_digit_accuracy(testresult, testlabels, NavieDic, trainingset):
	digit = np.zeros((10))
	correct_digit = np.zeros((10))

	confusion = np.zeros((11, 11))
	confusion[0, 1:] = np.arange(10)
	confusion[1:, 0] = np.arange(10)

	for i in range(len(testresult)):
		digit[testlabels[i]] += 1
		if testresult[i] == testlabels[i]:
			correct_digit[testlabels[i]] += 1

		confusion[testlabels[i]+1][testresult[i]+1] += 1

	print("Classification Rate for Digits 0 ~ 9:")
	for i in range(10):
		print(i, ": ", "{0:.2f}".format(100*correct_digit[i]/digit[i]), "%", end = "  ")
		if i == 3 or i == 6:
			print()
	print()
	confusion_mtx(testresult, testlabels, digit, confusion, NavieDic, trainingset)

def confusion_mtx(testresult, testlabels, digit, confusion, NavieDic, trainingset):
	for i in range(1, 11):
		confusion[i, 1:] /= digit[i-1]
	print()
	np.set_printoptions(precision=2)
	print(confusion)
	print()

	confusion = confusion[1:, 1:]

	ratiolist = []
	for i in range(20):
		row = np.argmax(np.max(confusion, axis=1))
		col = np.argmax(np.max(confusion, axis=0))
		ratiolist.extend([(row, col)])
		confusion[row, col] = 0
	ratiolist = ratiolist[10:14]
	print("With Laplace = 0.5, Top 4 Pairs with Highest Odd_Ratio:")
	print(ratiolist[0:4])

	# ratio(NavieDic, ratiolist, trainingset)

def ratio(NavieDic, ratiolist, trainingset):
	for tup in ratiolist:
		print(tup)
		oddratiomtx = np.zeros((28, 28))
		mat1 = np.zeros((28,28))
		mat2 = np.zeros((28,28))
		num1 = tup[0]
		num2 = tup[1]
		for i in range(28):
			for j in range(28):
				loc = (i, j)
				num1count = (NavieDic[loc]['+'][num1] + NavieDic[loc]['#'][num1]+1)/trainingset[num1]
				num2count = (NavieDic[loc]['+'][num2] + NavieDic[loc]['#'][num2]+1)/trainingset[num2]
				probability = num1count/num2count
				oddratiomtx[i, j] = probability
				mat1[i,j] = num1count
				mat2[i,j] = num2count

		heatmap(mat1, likelihood=True)
		heatmap(mat2, likelihood=True)
		heatmap(oddratiomtx)

def heatmap(matrix, likelihood=False):
	if likelihood:
		partitions = [.25, .5, .75]
	else:
		nmatrix = sorted(matrix.flatten())
		part_val = 28*28/5
		partitions = [nmatrix[int(part_val)], nmatrix[int(part_val*2)]
					, nmatrix[int(part_val*3)]]
	# import IPython
	# IPython.embed()
	# exit()
	for row in matrix:
		for val in row:
			if val < partitions[0]:
				print('@', end='')
			elif val > partitions[0] and val < partitions[1]:
				print('+', end='')
			elif val > partitions[1] and val < partitions[2]:
				print('-', end='')
			else:
				print('.', end='')
		print()
	print()


#=============================Binary Features=====================================

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

# P(piror) = trainingprior[i]
# Number of class[i] = trainingset[i]

def NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, laplace_const, pos_low, pos_high, testlabels):

	pos_low_value = np.zeros(10)
	pos_low_value[:] = np.inf
	pos_high_value = np.zeros(10)
	pos_high_value[:] = -np.inf

	for c in range(int(len(content)/28)):
		actual_label = testlabels[c]
		posteriori = [np.log(trainingprior[i]) for i in range(10)]
		for row in range(c*28, (c+1)*28):
			for col in range(28):
				loc = (row%28, col) 			# location of that grid
				cur_char = content[row][col]	# character in that grid

				classlist = NavieDic[loc][cur_char]

				for i in range(10):
					total_num = trainingset[i]
					lkhood = classlist[i]
					if lkhood == 0:
						lkhood = laplace_const
						total_num += laplace_const * 2

					posteriori[i] += np.log(lkhood/total_num)
		max_value = max(posteriori)
		max_label = posteriori.index(max(posteriori))

		if max_label == actual_label:
			if(pos_low_value[max_label] > max_value):
				pos_low_value[max_label] = max_value
				pos_low[max_label] = c
			if(pos_high_value[max_label] < max_value):
				pos_high_value[max_label] = max_value
				pos_high[max_label] = c
		testresult.extend([max_label])
		
	return pos_high, pos_low

#=============================Ternary Features=====================================

# def Building(content, traininglabels, NavieDic):
# 	for i in range(int(len(content)/28)):
# 		cur_label = traininglabels[i]
# 		for row in range(i*28, (i+1)*28):
# 			for col in range(28):
# 				loc = (row%28, col) 			# location of that grid
# 				cur_char = content[row][col]	# character in that grid
# 				if cur_char == '+':
# 					cur_char = '#'
# 				if loc not in NavieDic:			# not in the dictionary yet
# 					NavieDic.setdefault(loc, {})
# 					NavieDic[loc][' '] = np.zeros((10))
# 					NavieDic[loc]['#'] = np.zeros((10))
# 					NavieDic[loc][cur_char][cur_label] += 1
# 				else:
# 					NavieDic[loc][cur_char][cur_label] += 1

# # P(piror) = trainingprior[i]
# # Number of class[i] = trainingset[i]

# def NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, laplace_const, pos_low, pos_high):

# 	pos_low_value = np.zeros(10)
# 	pos_low_value[:] = np.inf
# 	pos_high_value = np.zeros(10)
# 	pos_high_value[:] = -np.inf

# 	for c in range(int(len(content)/28)):
# 		posteriori = [np.log(trainingprior[i]) for i in range(10)]
# 		for row in range(c*28, (c+1)*28):
# 			for col in range(28):
# 				loc = (row%28, col) 			# location of that grid
# 				cur_char = content[row][col]	# character in that grid
# 				if cur_char == '+':
# 					cur_char = '#'
# 				classlist = NavieDic[loc][cur_char]

# 				for i in range(10):
# 					total_num = trainingset[i]
# 					lkhood = classlist[i]
# 					if lkhood == 0:
# 						lkhood = laplace_const
# 						total_num += laplace_const * 2

# 					posteriori[i] += np.log(lkhood/total_num)
# 		max_value = max(posteriori)
# 		max_label = posteriori.index(max(posteriori))

# 		if(pos_low_value[max_label] > max_value):
# 			pos_low_value[max_label] = max_value
# 			pos_low[max_label] = c
# 		if(pos_high_value[max_label] < max_value):
# 			pos_high_value[max_label] = max_value
# 			pos_high[max_label] = c
# 		testresult.extend([max_label])
		
# 	return pos_high, pos_low
main()