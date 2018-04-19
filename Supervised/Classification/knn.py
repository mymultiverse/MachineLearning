from matplotlib import pyplot as plt  # install if you get no module error 
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
digits = datasets.load_digits()

#saperating train and test dataset
Train_X = iris.data[:120,:]
Train_Y = iris.target[:120]

Test_X = iris.data[120:,:]
Test_Y = iris.target[120:]

k=100 # hyperparameter for selecting number of the closest neighbours 

def pred(Train_X,Train_Y,Test_X,k):
	dist = np.zeros(shape=(30,120))
	pred_Y = np.zeros(shape=(30))
	for item in range(30):

		dist = np.sum(np.square(Train_X-Test_X[item,:]),axis=1)
		close_labels = Train_Y[dist.argsort()[:k]] # sorting the indexs of k mimimum distances and value at that index
		pred_Y[item] = np.bincount(close_labels).argmax() # choosing the most frequent label | counting frequency of each element and then max

	return pred_Y
