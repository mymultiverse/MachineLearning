from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
iris = datasets.load_iris()
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data) 