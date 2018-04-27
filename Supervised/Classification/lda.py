import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame()
df['x1']= [4,2,2,3,4,9,6,9,8,10]
df['x2']= [1,4,3,6,4,10,8,5,7,8]
df['class']=[1,1,1,1,1,2,2,2,2,2]
sns.set()

class1 = np.array([[4, 1], [2, 4],[2, 3],[3, 6],[4, 4]])
class2 = np.array([[9,10],[6, 8],[9, 5],[8, 7],[10, 8]])

m1 = np.mean(class1,axis=0)
m2 = np.mean(class2,axis=0)

scatter = {}

def scattr():
	l1= np.subtract(class1,m1)
	l2= np.subtract(class2,m2)
	s1 = np.zeros(shape=(2,2))
	s2 = np.zeros(shape=(2,2))
	scatter["s1"]=s1
	scatter["s2"]=s2

	for i, j in zip(l1,l2):
		i = np.reshape(i,(2,1))
		j = np.reshape(j,(2,1))
		s1+=i.dot(i.T)
		s2+=j.dot(j.T)
	Sw= s1+s2
	scatter["Sw"]=Sw
	return scatter 

scatter = scattr()

def weight(scatter):
	Sw = scatter["Sw"]
	W = np.dot(np.linalg.inv(Sw),m1-m2)
	return W

W = weight(scatter) 
