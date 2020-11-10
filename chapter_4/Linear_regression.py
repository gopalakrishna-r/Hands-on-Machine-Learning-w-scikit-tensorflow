import numpy as np

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1) # guassian noise

X_b = np.c_[np.ones((100,1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #moore's psuedo inverse

print(theta_best)

X_new = np.array([[0], [2]])
print(X_new)
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0,2, 0, 15])
# plt.show()

# Using the scikit
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best)