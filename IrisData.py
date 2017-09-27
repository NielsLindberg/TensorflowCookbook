from sklearn import datasets
iris = datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.target[0])
print(set(iris.target))

