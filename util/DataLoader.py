import os
import tarfile
import urllib.request

import pandas as pd

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/"


def fetch_gdp_life_satisfaction_data(datapath):
    os.makedirs(datapath, exist_ok=True)
    csv_li = "oecd_bli_2015.csv"
    csv_gdp = "gdp_per_capita.csv"
    print("Downloading", csv_li, csv_gdp)
    urllib.request.urlretrieve(DOWNLOAD_ROOT + "datasets/lifesat/" + csv_li, datapath + csv_li)
    urllib.request.urlretrieve(DOWNLOAD_ROOT + "datasets/lifesat/" + csv_gdp, datapath + csv_li)


TITANIC_PATH = os.path.join("..", "..", "datasets", "titanic-kaggle")


def fetch_titanic_data(file, titanic_data_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_data_path, file)
    return pd.read_csv(csv_path, sep=r'\s*,\s*', engine='python')


def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
