import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class1 = np.array([[4, 1], [2, 4],[2, 3],[3, 6],[4, 4]])
class2 = np.array([[9,10],[6, 8],[9, 5],[8, 7],[10, 8]])

m1 = np.mean(class1,axis=0)
m2 = np.mean(class2,axis=0)

scatter = {}
# calculation of scatter within class
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

W = weight(scatter).reshape((2,1))

#ploting decision boundary
df = pd.DataFrame()
df['x1']= [4,2,2,3,4,9,6,9,8,10]
df['x2']= [1,4,3,6,4,10,8,5,7,8]
df['class']=[1,1,1,1,1,2,2,2,2,2]

y=np.linspace(-6, 0).reshape((1,50))
fitx = np.dot(np.linalg.inv(np.dot(W, W.T)),np.dot(W,y)).T

fig = plt.subplots()

sns.set()
g = sns.lmplot(x="x1", y="x2", hue="class", data=df, fit_reg=False,  scatter_kws={"marker": "D","s": 70})
sns.regplot(fitx[:,0],fitx[:,1], scatter=False,line_kws={"color": "red"},ax=g.axes[0, 0],ci=None)
plt.title('Linear Discriminant Analysis')

#using sklearn
X = np.array([[4, 1],[2, 4],[2, 3],[3, 6],[4, 4],[9,10],[6, 8],[9, 5],[8, 7],[10, 8]])
Y = np.array([1,1,1,1,1,2,2,2,2,2])
clf = LinearDiscriminantAnalysis() #oject of model

clf.fit(X, Y)


h = .02     

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,aspect='auto', origin='lower')
colors = ['blue','green'] #color for class
plt.scatter(X[:, 0],X[:, 1], c=Y, cmap=matplotlib.colors.ListedColormap(colors))
# plt.plot(X[:, 0],X[:, 1], markersize=20)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xticks(())
plt.yticks(())
plt.title('Linear Discriminant Analysis')

plt.show()
