import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from util.MLUtil import build_transformer
from util.MLUtil import fetch_housing_data
from util.MLUtil import predict_with_best_model

pd.set_option('display.max_columns', None)

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/"
HOUSING_PATH = os.path.join("..", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

fetch_housing_data(HOUSING_URL, HOUSING_PATH)

csv_path = os.path.join(HOUSING_PATH, "housing.csv")

housing = pd.read_csv(csv_path)

# plot the data
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# split the data for testing
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(train_set.columns)

# create a new column to check whether train and test are representatives of the new columns
# i.e the after-effects of the splitting is represented in the sets.
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# plot the stratified samples
# housing["income_cat"].hist()
# plt.show()

# split the data into strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts(), len(strat_test_set))

# remove the newly added income category
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)

# copy the train set for exploring
housing = strat_train_set.copy()

# visualize the data

# The radius of each circle represents the district’s population (option s), and the color
# represents the price (option c). We will use a predefined color map (option cmap) called jet, which ranges from
# blue(low values) to red (high prices)
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100,
#              label="population", figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar="True",)
# plt.legend()
# plt.show()

# from pandas.plotting import scatter_matrix
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes],figsize=(12,8))
# plt.show()

# plot median income against median house value
# housing.plot(kind ="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# check for correlation and attribute combinations
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# preparing the data
# separating the labels from the train set
housing = strat_train_set.drop(["median_house_value"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# impute the empty values with median
# handle categorical values and apply transformation
housing_num = housing.drop("ocean_proximity", axis=1)
full_pipeline = build_transformer(housing_num)

housing_prepared = full_pipeline.fit_transform(housing)

print(pd.DataFrame(housing_prepared).columns)

# # train the model
# from sklearn.linear_model import LinearRegression
#
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
#
# some_data  = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
#
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions", lin_reg.predict(some_data_prepared))
# print("corresponding labels", list(some_labels))
#
# # check the error
# from sklearn.metrics import mean_squared_error
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_predictions,housing_labels)
# lin_rmse = np.sqrt(lin_mse)
#
# assert lin_rmse == 68628.19819848922
#
# from sklearn.tree import DecisionTreeRegressor
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
#
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_predictions, housing_labels)
# tree_rmse = np.sqrt(tree_mse)
#
# assert tree_rmse == 0.0
#
# # Evaluate with cross validation as the model is under-fitted
# from sklearn.model_selection import cross_val_score
#
# dec_tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# dec_tree_cross_rmse= np.sqrt(-dec_tree_scores)
#
# display_scores(dec_tree_cross_rmse)
#
# #Evaluate with cross validation for linear regression as decisiontree performed worse due to overfitting
#
#
# lin_reg_cross_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error",
# cv=10) lin_reg_cross_rmse= np.sqrt(-lin_reg_cross_scores)
#
# display_scores(lin_reg_cross_rmse)
#
# #Evaluate with cross validation for random forest to check if it will perform better than linear reg
#
# from sklearn.ensemble import RandomForestRegressor
#
# random_forest_reg = RandomForestRegressor()
# random_forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = random_forest_reg.predict(housing_prepared)
# ran_for_mse = mean_squared_error(housing_labels, housing_predictions)
# print("mean squared error", np.sqrt(ran_for_mse))
#
# ran_for_cross_scores = cross_val_score(random_forest_reg, housing_prepared, housing_labels,
# scoring="neg_mean_squared_error", cv=10) ran_for_cross_rmse= np.sqrt(-ran_for_cross_scores)
#
# display_scores(ran_for_cross_rmse)

# fine tune the model with grid search cv

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print("best parameters out of grid search :", grid_search.best_params_)

# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(f"mean absolute score {np.sqrt(-mean_score)} for parameter {params}")

# feature_importances = grid_search.best_estimator_.feature_importances_
# print(f"best features {feature_importances}")
#
# extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#
# print(sorted(zip(feature_importances, attributes), reverse= True))

# evaluate system on test set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

print(f"performance stat {predict_with_best_model(X_test,y_test,full_pipeline, final_model)}")


