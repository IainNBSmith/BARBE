"""
This interface ensures that no matter the type of model put into BARBE it will have the predict
 method attached to it. Needs to check for differently named prediction methods and find which
 one works, then coerce all of them into the same prediction method.
"""
import numpy as np
import torch


class BlackBoxWrapper:
    def __init__(self, bbmodel):
        self._bbmodel = bbmodel
        self._model_type = None

    def _predict_scikit(self, X):
        return self._bbmodel.predict(X)

    def _predict_torch(self, X):
        return self._bbmodel(torch.from_numpy(X.values))[:, 0]

    def predict(self, X):
        pass
