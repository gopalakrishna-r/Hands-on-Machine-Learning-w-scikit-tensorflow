import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# display a feature instance
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# training a binary classifier for digit 5
y_train_5 = (y_train == 5)  # true for all 5 and false for others. column would turn to true/false
y_test_5 = (y_test == 5)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42, n_jobs=10)

# using cross_val_predict for generating confusion matrix
from sklearn.model_selection import cross_val_predict

y_prob_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                  method="predict_proba")  # retrieves the probabilities instead of decision scores
y_score_forest = y_prob_forest[:, 1]  # score = proba of positive class

# plot ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_score_forest)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=None)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.show()

from sklearn.metrics import roc_auc_score

print(f"roc auc score : {roc_auc_score(y_train_5, y_score_forest)}")
