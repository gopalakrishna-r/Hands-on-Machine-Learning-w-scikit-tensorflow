from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784', version=1)
X,  y = mnist["data"], mnist["target"]

# display a feature instance
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

# plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000] , X[60000:], y[:60000], y[60000:]

# training a binary classifier for digit 5
y_train_5 = (y_train == 5) # true for all 5 and false for others. column would turn to true/false
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(cross_val_score(sgd_clf,X_train, y_train_5, cv= 5 , scoring="accuracy"))

# using cross_val_predict for generationg confusion matrix

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5,y_train_pred))
#   true negative-------> [[53892   687]<-------- false positive
#   false negative------> [ 1891  3530]]<-------- true positive

# precision and recall
from sklearn.metrics import precision_score, recall_score

print(f"precision of the model is: {precision_score(y_train_5, y_train_pred)} recall score is :  {recall_score(y_train_5, y_train_pred)}")

# F1 score
from sklearn.metrics import f1_score
print(f" F1 score of the model is: {f1_score(y_train_5, y_train_pred)}")