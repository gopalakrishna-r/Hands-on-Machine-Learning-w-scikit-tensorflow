import pandas as pd
import os
import numpy as np
from util.MLUtil import *
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

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
print(len(train_set), len(test_set))

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# plot the stratified samples
# housing["income_cat"].hist()
# plt.show()

# split the data into stratas
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
housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
