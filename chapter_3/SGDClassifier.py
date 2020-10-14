from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import cross_val_score

from util.MLUtil import plot_precision_recall_vs_threshold

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

# using the decision function to make predictions instead predict method of the model
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

# fetch decision scores from cross_val_predict
y_scores = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print(f"precision : {precisions} recall score is :  {recalls}, thresholds {thresholds}")

# plt.figure(figsize=(8, 4))
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
# plt.xlabel("Threshold", fontsize=16)
# plt.legend(loc="upper left", fontsize=16)
# plt.ylim([0, 1])
# plt.xlim([-700000, 700000])
# plt.show()

# figure out the 90 percent precision
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
print(f"threshold for 90 percent precision {threshold_90_precision}")

y_train_pred_90 = (y_scores >= threshold_90_precision)

print(f"precision of the model after threshold change is: {precision_score(y_train_5, y_train_pred_90)} recall score "
      f"is :  {recall_score(y_train_5, y_train_pred_90)}")

# plot ROC curve
# from sklearn.metrics import roc_curve
#
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, linewidth=2, label = None)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.axis([0,1,0,1])
# plt.xlabel('False Positive Rate', fontsize=16)
# plt.ylabel('True Positive Rate', fontsize=16)
# plt.show()

from sklearn.metrics import roc_auc_score
print(f"roc auc score : {roc_auc_score(y_train_5, y_scores)}")