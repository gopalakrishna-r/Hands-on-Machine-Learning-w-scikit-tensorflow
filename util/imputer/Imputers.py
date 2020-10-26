import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)  # returns the most frequent of a particular column
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)  # fills the most frequent at empty rows
