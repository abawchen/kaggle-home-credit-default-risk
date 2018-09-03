import numpy as np
import os
from .ModelWrapper import ModelWrapper


class XGBWrapper(ModelWrapper):

    def _predict(self, filename, X):
        if not os.path.exists(filename):
            # preds = self.clf.predict_proba(X, ntree_limit=self.clf.best_ntree_limit)[:, 1]
            preds = self.clf.predict_proba(X)[:, 1]
            np.save(filename, preds)
            return preds
        return np.load(filename)
