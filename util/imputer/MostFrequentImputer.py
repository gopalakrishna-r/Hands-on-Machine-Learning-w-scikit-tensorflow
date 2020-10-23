from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MostFrequentImputer(BaseEstimator, TransformerMixin):
   def fit(self, X, y= None):
       self.most_frequent_ = pd.Series([X[c].value_count().index[0] for c in X], index = X.columns) # returns the most frequent of a particular column
       return self
   def tranform(self, X, y=None):
       return X.fillna(self.most_frequent_) #fills the most frequent at empty rows