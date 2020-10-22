import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_predict

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

sgd_clf = SGDClassifier(random_state=42, n_jobs=10) # trains ten binary class classifier and gets the class with highest score
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

# using the decision function to make predictions instead predict method of the model
some_digit_scores = sgd_clf.decision_function([some_digit])
print(f" decision scores for the digit : f{some_digit_scores}  highest score :{np.argmax(some_digit_scores)}\n")

# plot confusion matrix for predictions
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
# conf_mx = confusion_matrix(y_train_pred, y_train)
# plt.matshow(conf_mx, cmap = plt.cm.gray)
# plt.show()
#
# # divide each element by the row sum i.e number of the images in the corresponding class
# row_sums = conf_mx.sum(axis = 1, keepdims = True)
# norm_conf_mx = conf_mx/row_sums
#
# # fill the diagonals with zero to keep the errors
# # rows represent the actual class and columns represent predicted ones
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
# plt.show()

# plotting misclassified 3s and 5s
# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.figure(figsize=(8,8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()