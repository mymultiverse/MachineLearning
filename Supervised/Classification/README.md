# Generative Models
### NAÏVE BAYES
Using bayes rule and finding stochastic model for each class assuming independent features. Mean and variances used to fit the gaussian distribution to each class. Plot below shows the probability distributions with respect to two features of iris dataset.

Sepal Dimensions
![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/nb.png)

Petal Dimentsions
![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/nb2.png)


# Discriminative Models
### Support Vector Machines
SVM classification model with different kernel function. Accuracy for iris dataset 
```
{'rbf': 0.9866666666666667, 'linear': 0.9933333333333333, 'poly': 0.98, 'sigmoid': 0.04}
```
Radial basis function
![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/svm_rbf.png)

Linear
![linear](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/svm_lin.png)

Polynomial
![poly](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/svm_poly.png)

Sigmoid
![sigmoid](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/svm_sigmoid.png)

### Decision Tree
![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/dtree.png)

### k-NN (k-Nearest Neighbor) Method
This is very basic implementation done on iris dataset with tolat 150 samples divided into 80% training and 20% test dataset. Accuracy of 
this model observred with various values of hyperparameter. In this case the optimal performance is at k=4.

![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/knn.png)

### Linear Discriminent Analysis
![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/lda.png "title-1") 

Using sklearn for fitting and predincting data
![](https://github.com/mymultiverse/MachineLearning/blob/master/Supervised/Classification/sklda.png "title-2")

## Dependencies
* Python Packages
  * Numpy
  * Matplotlib
  * Sklearn
  * GraphViz
  * mpl_toolkits
  * scipy
  * pandas
  * seaborn


#### Reference
[Sklearn](http://scikit-learn.org/stable/index.html)



