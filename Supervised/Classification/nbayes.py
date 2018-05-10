import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


iris = datasets.load_iris()

model = GaussianNB()
model.fit(iris.data, iris.target)
model.predict(iris.data)

accuracy = model.score(iris.data,iris.target)
print(accuracy)

Setosa = iris.data[:50,:]
Versicolour = iris.data[50:100,:]
Virginica = iris.data[100:,:]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

S_mux01 = np.mean(Setosa[:,:2],axis=0)
S_sigmax01 = np.std(Setosa[:,:2],axis=0)
S_cov = np.diag(S_sigmax01**2)

Ver_mux01 = np.mean(Versicolour[:,:2],axis=0)
Ver_sigmax01 = np.std(Versicolour[:,:2],axis=0)
Ver_cov = np.diag(Ver_sigmax01**2)


Vir_mux01 = np.mean(Virginica[:,:2],axis=0)
Vir_sigmax01 = np.std(Virginica[:,:2],axis=0)
Vir_cov = np.diag(Vir_sigmax01**2)

x= np.sort(Setosa[:,0])
y= np.sort(Setosa[:,1])
x, y = np.meshgrid(x,y)
# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

xy = np.column_stack([x.flat, y.flat])
z = multivariate_normal.pdf(xy, mean=S_mux01, cov=S_cov)
z = z.reshape(x.shape)
ax.plot_surface(x,y,z,color='m',label='Setosa')


x= np.sort(Versicolour[:,0])
y= np.sort(Versicolour[:,1])
x, y = np.meshgrid(x,y)
xy = np.column_stack([x.flat, y.flat])

xy = np.column_stack([x.flat, y.flat])
z = multivariate_normal.pdf(xy, mean=Ver_mux01, cov=Ver_cov)
z = z.reshape(x.shape)
ax.plot_surface(x,y,z, color='r',label='Ver')


x= np.sort(Virginica[:,0])
y= np.sort(Virginica[:,1])
x, y = np.meshgrid(x,y)
xy = np.column_stack([x.flat, y.flat])

xy = np.column_stack([x.flat, y.flat])
z = multivariate_normal.pdf(xy, mean=Vir_mux01, cov=Vir_cov)
z = z.reshape(x.shape)
ax.plot_surface(x,y,z,color='c',label='Vir')


# distribution plot for other two features
# S_mux01 = np.mean(Setosa[:,2:],axis=0)
# S_sigmax01 = np.std(Setosa[:,2:],axis=0)
# S_cov = np.diag(S_sigmax01**2)

# Ver_mux01 = np.mean(Versicolour[:,2:],axis=0)
# Ver_sigmax01 = np.std(Versicolour[:,2:],axis=0)
# Ver_cov = np.diag(Ver_sigmax01**2)


# Vir_mux01 = np.mean(Virginica[:,2:],axis=0)
# Vir_sigmax01 = np.std(Virginica[:,2:],axis=0)
# Vir_cov = np.diag(Vir_sigmax01**2)

# x= np.sort(Setosa[:,2])
# y= np.sort(Setosa[:,3])
# x, y = np.meshgrid(x,y)
# # Need an (N, 2) array of (x, y) pairs.
# xy = np.column_stack([x.flat, y.flat])

# xy = np.column_stack([x.flat, y.flat])
# z = multivariate_normal.pdf(xy, mean=S_mux01, cov=S_cov)
# z = z.reshape(x.shape)
# ax.plot_surface(x,y,z,color='m',label='Setosa')

# x= np.sort(Versicolour[:,2])
# y= np.sort(Versicolour[:,3])
# x, y = np.meshgrid(x,y)
# xy = np.column_stack([x.flat, y.flat])

# xy = np.column_stack([x.flat, y.flat])
# z = multivariate_normal.pdf(xy, mean=Ver_mux01, cov=Ver_cov)
# z = z.reshape(x.shape)
# ax.plot_surface(x,y,z, color='r',label='Ver')


# x= np.sort(Virginica[:,2])
# y= np.sort(Virginica[:,3])
# x, y = np.meshgrid(x,y)
# xy = np.column_stack([x.flat, y.flat])

# xy = np.column_stack([x.flat, y.flat])
# z = multivariate_normal.pdf(xy, mean=Vir_mux01, cov=Vir_cov)
# z = z.reshape(x.shape)
# ax.plot_surface(x,y,z,color='c',label='Vir')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Learned Gaussian Distributions')
plt.show()
