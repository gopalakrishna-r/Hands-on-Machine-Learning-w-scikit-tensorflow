from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

X,  y = mnist["data"], mnist["target"]
print(X.shape, y.shape)

# display a feature instance
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit