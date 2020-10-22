import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

mnist = fetch_openml('mnist_784', version=1)
X,  y = mnist["data"], mnist["target"]

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000] , X[60000:], y[:60000], y[60000:]

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier( n_jobs=10)
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))


# evaluate with f1 score
y_train_predd = cross_val_predict(knn_clf, X_train,  y_multilabel, cv=3)
print(f"f1 score {f1_score(y_multilabel, y_train_predd, average = 'macro')}")

# evaluate with f1 score with weighted score
y_train_predd = cross_val_predict(knn_clf, X_train,  y_multilabel, cv=3)
print(f"f1 score {f1_score(y_multilabel, y_train_predd, average = 'weighted')}")