import numpy as np

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1) # guassian noise

X_b = np.c_[np.ones((100,1)), X] # add x0 = 1 to each instance

eta = 0.1 # learning rate
n_iterations = 1000
m = 100 # batch size

theta = np.random.randn(2, 1) # random initialization

for i in range(n_iterations):
    gradients = 2 / m * (X_b.T.dot(X_b.dot(theta) - y))
    theta = theta - gradients * eta

print(theta)