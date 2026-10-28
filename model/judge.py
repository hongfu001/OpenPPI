"""Fixed judge-score baseline with explicit feature columns."""
import numpy as np


class JudgeModel:
    def __init__(self, columns):
        self.columns = np.asarray(list(columns))
        if (self.columns.ndim != 1 or not len(self.columns)
                or self.columns.dtype.kind not in 'iu' or (self.columns < 0).any()
                or len(np.unique(self.columns)) != len(self.columns)):
            raise ValueError('Provide distinct nonnegative integer judge columns')

    def predict(self, features):
        x = np.asarray(features)
        if x.ndim != 2 or self.columns.max() >= x.shape[1]:
            raise ValueError('Judge columns must exist in the feature matrix')
        return x[:, self.columns].mean(axis=1)
