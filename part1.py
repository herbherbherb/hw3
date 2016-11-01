import numpy as np
from copy import copy, deepcopy
import time
from collections import defaultdict as setdefault

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

	poss_laplace = [0.1, 0.5, 1, 2, 3, 5, 10]
	for i in range(len(poss_laplace)):
		start_time = time.time()
		NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, poss_laplace[i])
		end = time.time()
		correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])
		Accuracy_rate = (correct_predicted/len(testlabels))*100
		print("Accuracy (Laplace = ", poss_laplace[i], "): ", Accuracy_rate, "%")
		print("Testing Time: ", end - start_time)
		testresult = []


					
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

def NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, laplace_const):
	for i in range(int(len(content)/28)):

		posteriori = [np.log(trainingprior[i]) for i in range(10)]
		for row in range(i*28, (i+1)*28):
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
		max_label = posteriori.index(max(posteriori))
		testresult.extend([max_label])

# def NavieClassify(content, NavieDic, testresult, trainingset, trainingprior, laplace_const):
# 	for i in range(int(len(content)/28)):

# 		posteriori = [np.log(trainingprior[i]) for i in range(10)]
# 		cur_set = deepcopy(trainingset)
# 		for row in range(i*28, (i+1)*28):
# 			for col in range(28):
# 				loc = (row%28, col) 			# location of that grid
# 				cur_char = content[row][col]	# character in that grid
			
# 				classlist = NavieDic[loc][cur_char]

# 				for i in range(10):
# 					lkhood = classlist[i]
# 					if lkhood == 0:
# 						lkhood = laplace_const
# 						cur_set[i] += laplace_const
# 					posteriori[i] += np.log(lkhood/cur_set[i])
# 		max_label = posteriori.index(max(posteriori))
# 		testresult.extend([max_label])
main()


# import numpy as np
# from copy import copy, deepcopy
# import time
# from collections import defaultdict as setdefault

# def main():
# #=======initialize================
# 	traininglabels = []
# 	testlabels = []
# 	trainingprior = np.zeros((10))
# 	likelihood = []
# 	testresult = []

# 	NavieDic = {}
# 	NavieDic_prior = {}
# #========Reading==================
# 	with open('traininglabels') as f:
# 		content = f.readlines()
# 	traininglabels = [int(x.strip('\n')) for x in content]

# 	with open('testlabels') as f:
# 		content = f.readlines()
# 	testlabels = [int(x.strip('\n')) for x in content]


# 	trainingset = {}						# P(piror) = trainingprior[i] # of class[i] = trainingset
# 	traininglength = len(traininglabels)
# 	for i in range(10):
# 		count = traininglabels.count(i)
# 		trainingset[i] = count
# 		trainingprior[i] = count/traininglength
# 		likelihood.append([])

# 	with open('trainingimages') as f:
# 		content = f.readlines()
		
# 	Building(content, traininglabels, NavieDic)
# 	Building_piror(content, traininglabels, NavieDic_prior)
	
# #===========Testing================================================
# 	with open('testimages') as f:
# 		content = f.readlines()

# 	poss_laplace = [0.1, 0.5, 1, 2, 5, 10]
# 	for i in range(len(poss_laplace)):
# 		NavieClassify(content, NavieDic, NavieDic_prior, testresult, trainingset, trainingprior, poss_laplace[i])
# 		correct_predicted = len([i for i, j in zip(testlabels, testresult) if i == j])
# 		Accuracy_rate = (correct_predicted/len(testlabels))*100
# 		print("Accuracy (Laplace = ", poss_laplace[i], "): ", Accuracy_rate, "%")
# 		testresult = []


# def Building_piror(content, traininglabels, NavieDic_prior):
# 	class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 	for i in range(int(len(content)/28)):
# 		cur_label = traininglabels[i]
# 		for row in range(i*28, (i+1)*28):
# 			for col in range(28):
# 				loc = (row%28, col) 			# location of that grid

# 				if loc not in NavieDic_prior:
# 					NavieDic_prior.setdefault(loc, {})
# 					for c in class_list:
# 						NavieDic_prior[loc][c] = 0
# 					NavieDic_prior[loc][cur_label] += 1
# 				else:
# 					NavieDic_prior[loc][cur_label] += 1

# def Building(content, traininglabels, NavieDic):
# 	for i in range(int(len(content)/28)):
# 		cur_label = traininglabels[i]
# 		for row in range(i*28, (i+1)*28):
# 			for col in range(28):
# 				loc = (row%28, col) 			# location of that grid
# 				cur_char = content[row][col]	# character in that grid
# 				if loc not in NavieDic:			# not in the dictionary yet
# 					NavieDic.setdefault(loc, {})
# 					NavieDic[loc][' '] = np.zeros((10))
# 					NavieDic[loc]['+'] = np.zeros((10))
# 					NavieDic[loc]['#'] = np.zeros((10))
# 					NavieDic[loc][cur_char][cur_label] += 1
# 				else:
# 					NavieDic[loc][cur_char][cur_label] += 1

# def NavieClassify(content, NavieDic, NavieDic_prior, testresult, trainingset, trainingprior, laplace_const):
# 	for i in range(int(len(content)/28)):

# 		posteriori = [np.log(trainingprior[i]) for i in range(10)]
# 		for row in range(i*28, (i+1)*28):
# 			for col in range(28):
# 				loc = (row%28, col) 			# location of that grid
# 				cur_char = content[row][col]	# character in that grid
			
# 				classlist = NavieDic[loc][cur_char]

# 				for i in range(10):
# 					total_num = NavieDic_prior[loc][i]
# 					lkhood = classlist[i]
# 					if lkhood == 0:
# 						lkhood = laplace_const
# 						total_num += laplace_const * 2

# 					posteriori[i] += np.log(lkhood/total_num)
# 		max_label = posteriori.index(max(posteriori))
# 		testresult.extend([max_label])

# main()