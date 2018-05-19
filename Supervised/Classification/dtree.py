import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
model = tree.DecisionTreeClassifier()
model = model.fit(iris.data, iris.target)

# tree generation considering all features 
dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Decision Tree for iris dataset")

model.predict(iris.data)

# analysis considering two features 
labels = 3
plot_colors = "ryb"
plot_step = 0.02

for index, features in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # for visualise in 2d taking only 2 feature at once
    #for taining, prediction and plot

    X = iris.data[:, features]
    Y = iris.target

    clf = tree.DecisionTreeClassifier().fit(X, Y)

    #decision boundary
    plt.subplot(2, 3, index + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[features[0]])
    plt.ylabel(iris.feature_names[features[1]])

    # Plot the training points
    for i, color in zip(range(labels), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using two features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()
