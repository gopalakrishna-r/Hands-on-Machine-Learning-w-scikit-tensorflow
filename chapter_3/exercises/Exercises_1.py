import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score

# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
# on the test set. Hint: the KNeighborsClassifier works quite well for this task;
# you just need to find good hyperparameter values (try a grid search on the
# weights and n_neighbors hyperparameters).

mnist = fetch_openml('mnist_784', version=1)
X,  y = mnist["data"], mnist["target"]

y = y.astype(np.uint8)

# splitting data
X_train, X_test, y_train, y_test = X[:60000] , X[60000:], y[:60000], y[60000:]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights':['uniform', 'distance'],
              'n_neighbors':[3, 4, 5]}]

knn_clf = KNeighborsClassifier()

grid_search = GridSearchCV(knn_clf,param_grid, cv = 5, n_jobs=-1 ,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,verbose=3)
grid_search.fit(X_train, y_train)
print(f"best parameter values {grid_search.best_params_}") # {'n_neighbors': 4, 'weights': 'distance'}

best_model = grid_search.best_estimator_

print(f"accuracy {cross_val_score(best_model,X_train,y_train, cv=5, scoring='accuracy')}")