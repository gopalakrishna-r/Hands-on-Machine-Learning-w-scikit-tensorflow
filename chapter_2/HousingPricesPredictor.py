import pandas as pd
import os
from util.MLUtil import fetch_housing_data


DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/"
HOUSING_PATH = os.path.join("..", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

fetch_housing_data(HOUSING_URL,HOUSING_PATH)

csv_path = os.path.join(HOUSING_PATH, "housing.csv")

housingInfo = pd.read_csv(csv_path)
print(housingInfo.head(5))
