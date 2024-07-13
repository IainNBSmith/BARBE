"""
This interface ensures that no matter the type of model put into BARBE it will have the predict
 method attached to it. Needs to check for differently named prediction methods and find which
 one works, then coerce all of them into the same prediction method.
"""
import numpy as np
import torch


class BlackBoxWrapper:
    def __init__(self, bbmodel, class_input=None):
        self._bbmodel = bbmodel
        self._class_binary = class_input
        self._model_type = None
        self._assign_type()

    def _assign_type(self):
        pred_function = getattr(self._bbmodel, "predict", None)
        if callable(pred_function):
            self._model_type = 'sklearn-like'
            return None
        if callable(self._bbmodel):
            self._model_type = 'torch-like'
            return None
        assert ValueError("Input model does not contain a model.predict(X) function or is not itself callable. "
                          "Have you run model.eval()?")

    def set_class(self, label):
        self._class_binary = label

    def _binary_assignment(self, y):
        if self._class_binary is None:
            return y
        else:
            return y == self._class_binary

    def _predict_scikit(self, X):
        return self._bbmodel.predict(X)

    def _predict_torch(self, X):
        return self._bbmodel(torch.from_numpy(X.values))[:, 0]

    def predict(self, X):
        if self._model_type == 'sklearn-like':
            return self._binary_assignment(self._predict_scikit(X))
        elif self._model_type == 'torch-like':
            return self._binary_assignment(self._predict_torch(X))
        return None
