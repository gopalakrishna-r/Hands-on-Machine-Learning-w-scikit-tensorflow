from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import cross_val_score

from util.MLUtil import plot_precision_recall_vs_threshold

mnist = fetch_openml('mnist_784', version=1)
X,  y = mnist["data"], mnist["target"]

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000] , X[60000:], y[:60000], y[60000:]

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
print(forest_clf.predict([some_digit]))

# using the decision function to make predictions instead predict method of the model
some_digit_scores = forest_clf.predict_proba([some_digit])
print(f" prediction probabilities for the digit : f{some_digit_scores}  highest score :{np.argmax(some_digit_scores)}\n")

print(f"accuracy {cross_val_score(forest_clf,X_train,y_train, cv=5, scoring='accuracy')}")