from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict

from util.MLUtil import plot_precision_recall_vs_threshold

mnist = fetch_openml('mnist_784', version=1)
X,  y = mnist["data"], mnist["target"]

# display a feature instance
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[1]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000] , X[60000:], y[:60000], y[60000:]

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42) # trains ten binary class classifier and gets the class with highest score
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

# using the decision function to make predictions instead predict method of the model
some_digit_scores = sgd_clf.decision_function([some_digit])
print(f" decision scores for the digit : f{some_digit_scores}  highest score :{np.argmax(some_digit_scores)}\n")

# plot confusion matrix for predictions
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train_pred, y_train)
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()