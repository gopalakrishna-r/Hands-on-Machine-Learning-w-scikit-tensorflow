import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
# on the test set. Hint: the KNeighborsClassifier works quite well for this task;
# you just need to find good hyperparameter values (try a grid search on the
# weights and n_neighbors hyperparameters).
from util.MLUtil import shift_image

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

X_train_aug = [image for image in X_train]
y_train_aug = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_aug.append(shift_image(image, dx, dy))
        y_train_aug.append(label)

X_train_aug = np.array(X_train_aug)
y_train_aug = np.array(y_train_aug)

shuffle_ind = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuffle_ind]
y_train_aug = y_train_aug[shuffle_ind]

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4, n_jobs=-1)  # best parameters from the first exercise

knn_clf.fit(X_train_aug, y_train_aug)

y_pred = knn_clf.predict(X_test)

print(f"accuracy {accuracy_score(y_test, y_pred)}")
