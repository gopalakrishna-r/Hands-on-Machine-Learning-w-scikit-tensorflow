import pandas as pd
import os
import tarfile
from six.moves import urllib
import numpy as np

import urllib.request

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