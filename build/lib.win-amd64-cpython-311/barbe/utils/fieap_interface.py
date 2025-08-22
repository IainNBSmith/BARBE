# TODO: add code that trains + validates the ensemble
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from FAPFID.fapfid import FAPFID_algorithm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from barbe.discretizer import CategoricalEncoder
from itertools import product


class FIEAPClassifier:
    def __init__(self,
                 protected_feature=None,
                 privileged_group=None,
                 unprivileged_group=None,
                 base_classifier=DecisionTreeClassifier(random_state=42),
                 num_clusters=4):
        self.encoder = CategoricalEncoder(ordinal_encoding=False)
        self.diff_encoder = CategoricalEncoder(ordinal_encoding=False)
        self.training_data = None

        self._le = LabelEncoder()
        self._ensemble_models = None
        self._meta_model = None
        self._meta_data = None
        self._meta_settings = {'privileged_group': privileged_group,
                               'unprivileged_group': unprivileged_group,
                               'protected_feature': protected_feature,
                               'base_classifier': base_classifier,
                               'num_clusters': num_clusters}

    def fit(self, X, y):
        self.training_data = X.copy()
        self.diff_encoder.fit(X.copy())
        X = self.encoder.fit_transform(training_data=X)
        y = self._le.fit_transform(y)

        rf_criterion = ['gini']
        rf_depth = [2, 5, 10, None]
        rf_max_features = ['sqrt']
        svc_kernel = ['rbf', 'poly', 'sigmoid']
        svc_c = [0.2, 0.5, 1.0]
        xg_depth = [2, 5, 10]
        xg_lambda = [1]
        lr_penalty = [None]
        lr_c = [0.2, 0.5, 1.0]
        hyperparams = [rf_criterion, rf_depth, rf_max_features,
                       svc_kernel, svc_c,
                       xg_depth, xg_lambda,
                       lr_penalty, lr_c]
        X['target'] = y
        X_train, X_test = train_test_split(X, train_size=0.8, random_state=83, stratify=X['target'])
        best_accuracy = 0
        for params in product(*hyperparams):
            print(params)
            rf_crit, rf_dep, rf_features, svc_k, svc_c_val, xg_dep, xg_lam, lr_pen, lr_c_val = params
            ensemble_models = [RandomForestClassifier(criterion=rf_crit, max_depth=rf_dep, max_features=rf_features),
                               SVC(kernel=svc_k, C=svc_c_val, probability=True),
                               XGBClassifier(max_depth=xg_dep, reg_lambda=xg_lam, enable_categorical=True),
                               LogisticRegression(penalty=lr_pen, C=lr_c_val)]
            print("FAPFID Running...")
            meta_model, meta_data = FAPFID_algorithm(data=X_train,
                                                     ensemble_models=ensemble_models,
                                                     **self._meta_settings)
            curr_accuracy = accuracy_score(X_test['target'], meta_model.predict(meta_data(X_test.drop('target', axis=1))))
            print(curr_accuracy)
            print(confusion_matrix(X_test['target'], meta_model.predict(meta_data(X_test.drop('target', axis=1)))))
            if curr_accuracy > best_accuracy:
                self._ensemble_models = ensemble_models.copy()
                best_accuracy = curr_accuracy

        print("BEST ACCURACY: ", best_accuracy)
        self._meta_model, self._meta_data = FAPFID_algorithm(data=X,
                                                             ensemble_models=self._ensemble_models,
                                                             **self._meta_settings)

    def _inner_predict(self, X):
        # returns all each of the 4 predictions from inner models
        meta_data = np.array([model.predict_proba(X)[:, 0] for model in self._ensemble_models]).T
        return meta_data

    def predict(self, X):
        X = self.encoder.transform(X)
        meta_data = self._inner_predict(X)
        return self._le.inverse_transform(self._meta_model.predict(meta_data))
