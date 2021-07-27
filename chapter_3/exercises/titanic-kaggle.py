#Tackle the Titanic dataset
#The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on.

import os

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from util.DataLoader import fetch_titanic_data

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

TITANIC_PATH = '..\\..\\datasets\\titanic-kaggle'

for dirname, _, filenames in os.walk('../datasets/titanic-kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# loading data
train_data = fetch_titanic_data("train.csv")
test_data = fetch_titanic_data("test.csv")
# ### Visualizing data

print(train_data.head())
print(train_data.info())
# Null values in age, embarked, and cabin. Ignoring cabin as it largely consists of null values.

print(train_data.describe())

print(
    f"values counts of  survived :{train_data['Survived'].value_counts()},\n Pclass {train_data['Pclass'].value_counts()} \n,"
    f" Sex : {train_data['Sex'].value_counts()},\n Embarked :{train_data['Embarked'].value_counts()}\n")

# ### Creating labels
train_y, train_X = train_data['Survived'], train_data.drop(['Survived'], axis=1),
test_X = test_data

# ### check for cardinality and unique categories in test data

# get list of categorical variables
s = (train_X.dtypes == 'object')
categorical_cols = list(s[s].index)

# Columns that can be safely label encoded
good_label_cols = [col for col in categorical_cols if
                   set(train_X[col]) == set(test_X[col])]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(categorical_cols) - set(good_label_cols))

print(f"columns without unseen values{good_label_cols} \n with unseen values {bad_label_cols}")
# Most of the category columns in the test data have values not present in the train data
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: train_X[col].nunique(), categorical_cols))
d = dict(zip(categorical_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

print(f"value counts of the categories {d}")

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in categorical_cols if train_X[col].nunique() < 15]
print(f"low cardinality columns {low_cardinality_cols}")

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(categorical_cols) - set(low_cardinality_cols))
print(f"high cardinality columns {high_cardinality_cols}")

# Select numerical columns
numerical_cols = [cname for cname in train_X.columns if
                  train_X[cname].dtype in ['int64', 'float64']]
numerical_cols.remove('PassengerId')
print(f"numerical columns {numerical_cols}")

# join both categorical and numerical pipeline
cols = categorical_cols + numerical_cols

# ### null/missing values handling and pipeline creation
# #### numeric pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

# #### category pipeline
from sklearn.compose import ColumnTransformer

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ("onehot", OneHotEncoder(sparse=False, handle_unknown='ignore')), ])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

train_X_prepared = train_X[cols]

# ### creating model
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

forest_clf = RandomForestClassifier(n_estimators=1200, max_features='sqrt', max_depth=60, bootstrap=False)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', forest_clf)])

# fit the pipeline and predict
pipeline.fit(train_X_prepared, train_y)

predictions = pipeline.predict(test_X)

output = pd.DataFrame({
    "PassengerId": test_X["PassengerId"],
    "Survived": predictions
})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
