import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def indices_of_features(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_imortances, k):
        self.features_importances = feature_imortances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices = indices_of_features(self.features_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices]


# A class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]
