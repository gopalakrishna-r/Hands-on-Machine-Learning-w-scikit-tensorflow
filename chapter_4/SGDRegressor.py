from sklearn.linear_model import SGDRegressor
import numpy as np

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1) # guassian noise

sgd_reg = SGDRegressor(max_iter=1000, tol = 1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

print(f"intercept: {sgd_reg.intercept_}, slope : {sgd_reg.coef_}")