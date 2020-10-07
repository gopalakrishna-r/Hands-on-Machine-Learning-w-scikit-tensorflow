from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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
