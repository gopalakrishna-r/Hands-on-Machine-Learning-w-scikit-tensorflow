from scipy.stats import reciprocal, expon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

from util.MLUtil import *
from util.MLUtil import build_transformer

pd.set_option('display.max_columns', None)

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/"
HOUSING_PATH = os.path.join("..", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

fetch_housing_data(HOUSING_URL, HOUSING_PATH)

csv_path = os.path.join(HOUSING_PATH, "housing.csv")

housing = pd.read_csv(csv_path)

# split the data for testing
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# create a new column to check whether train and test are representatives of the new columns
# i.e the after-effects of the splitting is represented in the sets.
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# split the data into strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# remove the newly added income category
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)

# copy the train set for exploring
housing = strat_train_set.copy()

# check for correlation and attribute combinations
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# preparing the data
# separating the labels from the train set
housing = strat_train_set.drop(["median_house_value"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# impute the empty values with median
# handle categorical values and apply transformation
housing_num = housing.drop("ocean_proximity", axis=1)
full_pipeline = build_transformer(housing_num)

housing_prepared = full_pipeline.fit_transform(housing)

n_grid = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm = SVR()

random_search = RandomizedSearchCV(svm, cv=5, n_iter=50, verbose=2,
                                   scoring='neg_mean_squared_error', param_distributions=n_grid,
                                   return_train_score=True, n_jobs=10)
random_search.fit(housing_prepared, housing_labels)

print("best parameters out of grid search :", random_search.best_params_) # {'C': 151308.41574042197, 'gamma': 0.24624812365129908, 'kernel': 'rbf'} final prediction score 52514.959533442394

# evaluate system on test set

final_model = random_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

print(f"performance stat {predict_with_best_model(X_test, y_test, full_pipeline, final_model)}")
