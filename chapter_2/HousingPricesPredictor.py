from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from util.MLUtil import *
from util.transformer import CombinedAttributesAdder

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
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# handle categorical values
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head())

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_encoded = cat_encoder.fit_transform(housing_cat)

# creating custom transformer
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)