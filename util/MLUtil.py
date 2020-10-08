import pandas as pd
import os
import tarfile
from six.moves import urllib
import numpy as np
from sklearn.metrics import mean_squared_error
import urllib.request

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

from util.TopFeatureSelector import TopFeatureSelector

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/"


def fetch_gdp_life_satisfaction_data(datapath):
    os.makedirs(datapath, exist_ok=True)
    csv_li = "oecd_bli_2015.csv"
    csv_gdp = "gdp_per_capita.csv"
    print("Downloading", csv_li, csv_gdp)
    urllib.request.urlretrieve(DOWNLOAD_ROOT + "datasets/lifesat/" + csv_li, datapath + csv_li)
    urllib.request.urlretrieve(DOWNLOAD_ROOT + "datasets/lifesat/" + csv_gdp, datapath + csv_li)


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


from zlib import crc32


# determine whether the test data instance's identifier falls within the ratio
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) ^ 0xfffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    print(in_test_set)
    return data.iloc[~in_test_set], data.iloc[in_test_set]


def display_scores(scores):
    print("scores:", scores)
    print("mean:", scores.mean())
    print("standard deviation", scores.std())


def build_transformer(housing_numericals, cat_attribs=None):
    if cat_attribs is None:
        cat_attribs = ["ocean_proximity"]
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from util.transformer import CombinedAttributesAdder
    num_attribs = list(housing_numericals)

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])
    return full_pipeline


def build_transformer_with_features_model(housing_numericals, feature_count, top_features=None, cat_attribs=None,
                                          best_parameters=None):
    pipeline = build_transformer(housing_numericals, cat_attribs)
    pipeline_steps_ = [
        ('preparation', pipeline)
    ]
    if top_features is None:
        if best_parameters is None:
            pass
        else:
            pipeline_steps_ = pipeline_steps_.append([('svm_reg', SVR(best_parameters))])
    else:
        if best_parameters is None:
            pipeline_steps_ = pipeline_steps_.append([('feature_selection', TopFeatureSelector(top_features, feature_count))])
        else:
            pipeline_steps_ = pipeline_steps_.append([('svm_reg', SVR(best_parameters)),
                               ('feature_selection', TopFeatureSelector(top_features, feature_count))])

    return Pipeline(pipeline_steps_)


def predict_with_best_model(X_test, y_test, full_pipeline, final_model):
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"final prediction score {final_rmse}")
    # checking the precision of the model using confidence level
    from scipy import stats
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    performance_stat = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                                loc=squared_errors.mean(),
                                                scale=stats.sem(squared_errors)))
    return performance_stat
