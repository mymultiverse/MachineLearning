from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
digits = datasets.load_digits()
Train_X = iris.data[:120,:]
Train_Y = iris.target[:120]

Test_X = iris.data[120:,:]
Test_Y = iris.target[120:]



	# dist = np.append(dist, d, axis=0)

def pred(Train_X,Train_Y,Test_X,k):

	dist = np.zeros(shape=(30,120))

	for item in range(30):

		dist = np.sum(np.square(Train_X-Test_X[item,:]),axis=1)
		
		# dist[item,:] = np.sum(np.square(Train_X-Test_X[item,:]),axis=1)
		mini_idx =  dist.argsort()[:k]
		close_labels = Train_Y[mini_idx]




	# mini_idx = np.argmin(dist, axis=1)



	return close_labels
			
l=np.array([2,6,8,9,0,12])
a = np.array([3,2,3,1,2,1,3,1,3,2,2,3])
print np.bincount(a).argmax()
