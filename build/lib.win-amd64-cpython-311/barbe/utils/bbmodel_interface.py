"""
This interface ensures that no matter the type of model put into BARBE it will have the predict
 method attached to it. Needs to check for differently named prediction methods and find which
 one works, then coerce all of them into the same prediction method.
"""
import numpy as np
import pandas as pd
import torch


class BlackBoxWrapper:
    __doc__ = '''
    Purpose: Utility for BARBE, wraps black box models to give them a common call using obj.predict() and adds method
     for changing classification to binary labels rather than multiclass.
     
    Input: bbmodel (object or callable)         -> trained method that makes a prediction on data.
           class_input (return type of bbmodel) -> class used for binary output (will be True and other labels False).
            | Default: None
    '''

    def __init__(self, bbmodel, class_input=None, class_labels=None):
        self._bbmodel = bbmodel
        self._class_binary = class_input
        self._class_labels = class_labels
        self._model_type = None
        self._assign_type()

    def _assign_type(self):
        """
        Purpose: Check that there is something like a predict function in the given bbmodel. Based on the available
         functions (itself callable or a callable predict function) assign it a label that will be used later.
        """
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
        """
        Purpose: Set class used in binary classification e.g. bbmodel predicts 1, 2, 3, ..., 9 if _class_binary = 2
         then this will return False, True, False, ..., False instead.
        """
        self._class_binary = label

    def _binary_assignment(self, y):
        """
        Input: y (1d numpy array) -> classifications to check
        Purpose: Convert predictions that may have multiple labels into a binary True or False.
        Output: Either y or an array of True and False values according to the binary class.
        """
        if self._class_binary is None:
            return y.astype(str)
        else:
            return np.array([str(self._class_binary[0]) if yi == self._class_binary
                             else '~' + str(self._class_binary[0]) for yi in y])

    def _split_binary_assignment(self, y):
        """
        Input: y (1d numpy array) -> classifications to check
        Purpose: Convert predictions that may have multiple labels into a binary True or False.
        Output: Either y or an array of True and False values according to the binary class.
        """
        distinct = self._class_labels
        out_y = np.ndarray(shape=(len(y), len(distinct)))
        for i in range(len(distinct)):
            label = distinct[i]
            out_y[y == label, i] = 1
            out_y[y != label, i] = 0

        return out_y

    def _predict_scikit(self, X):
        #if X.shape[0] == 1:
        #    X = pd.concat([X, X], ignore_index=True)
        #    return [self._bbmodel.predict(X)[0]]
        return self._bbmodel.predict(X)

    def _predict_torch(self, X):
        # IAIN must also detach I think
        return self._bbmodel(torch.from_numpy(X.values))[:, 0]

    def predict_proba(self, X):
        return self._split_binary_assignment(self._predict_scikit(X))
    def predict(self, X):
        if self._model_type == 'sklearn-like':
            return self._binary_assignment(self._predict_scikit(X))
        elif self._model_type == 'torch-like':
            return self._binary_assignment(self._predict_torch(X))
        return None

    def check_valid_data(self, X):
        try:
            self.predict(X)
            return 1
        except ValueError:
            return 0
