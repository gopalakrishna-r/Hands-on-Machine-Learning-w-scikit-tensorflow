import numpy as np


np.random.seed(0)
probs = np.array([0.1, 0.0, 0.7, 0.2])
idx = np.random.choice(range(len(probs)), p = probs)
print(idx)
