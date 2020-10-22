# A system which removes noise from the images
# Outputs a clean image which consists array of pixel intensities
# Output is multilabel(pixel). The pixel can have multiple values(intensities from 0 to 255)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

from util.MLUtil import plot_digit

mnsit = fetch_openml('mnist_784',version=1 )
X, y = mnsit['data'], mnsit['target']

X_train,X_test,y_train,y_test = X[:60000], X[60000:], y[:60000] , y[60000:]

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


#plot a noisy image
some_digit = X_test_mod[2] # five with noise
some_digit_image = some_digit.reshape(28,28)
#
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_jobs=10)
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([some_digit])
plot_digit(clean_digit)