# θ = (XTX + αA)−1 * XT * y
import numpy as np
from sklearn.linear_model import Ridge

from util.MLUtil import plot_learning_curve

m = 100
X = 6 * np.random.rand(m, 1) -3
y = 0.5 * X ** 2 + X + 2 + np.random.rand(m, 1) # A quadratic equation is of the form y = ax2 + bx + c.

ridge_reg = Ridge(alpha= 1, solver="cholesky")

plot_learning_curve(ridge_reg, X, y)

