# Ensure importing works correctly
import numpy as np
import numpy.random
import pandas as pd

from barbe.explainer import BARBE
from barbe.perturber import BarbePerturber, ClassBalancedPerturber
from barbe.discretizer import CategoricalEncoder
from barbe.counterfactual import BarbeCounterfactual

from barbe.utils.sigdirect_interface import SigDirectWrapper
from numpy import random


class DummyBlackBox:
    def __init__(self, mode="unbiased", label_type="numeric", n_important_dims=3):
        self._rng = np.random.default_rng()
        self._types = None
        self._importance = None
        self._modes = None
        self._n_important_dims = n_important_dims
        self._labels = [-1, 1] if label_type == "numeric" else ["A", "B"]
        self._odds = 0 if mode == "unbiased" else 0.5

    def _mode(self, X):
        mode_list = list()
        for column_name in X.columns:
            unique_entries = np.unique(X[column_name].astype(str))
            print(X[column_name].dtypes)
            if len(unique_entries) > 10 and X[column_name].dtypes != "object":
                mode_list.append(np.nanmedian(X[column_name].astype(float)))
            else:
                best_count = 0
                best_item = None
                for item in unique_entries:
                    item_count = np.sum(X[column_name] == item)
                    if item_count > best_count:
                        best_item = item
                        best_count = item_count
                mode_list.append(best_item)
        return mode_list

    def fit(self, X, y):
        # get the data format
        self._types = [X[X.columns[i]].dtypes for i in range(X.shape[1])]  # for dataframe
        # pick random dimension to be important
        n_dims_x = X.shape[1]
        self._importance = self._rng.standard_cauchy(size=n_dims_x) + self._odds
        print(self._importance)
        self._modes = self._mode(X)  # for numbers within +/- 1 of mode for string just mode
        print(self._modes)

    def predict_dummy(self, X):
        # check data format (order, type, etc.)
        # return random labels based on the mode (more of one or balanced)
        pred_y = None
        for i in range(X.shape[1]):
            if self._types[i] == 'object':
                if pred_y is None:
                    pred_y = X[X.columns[i]].apply(func=lambda x: 1 if x == self._modes[i] else -1) * self._importance[i]
                else:
                    pred_y += X[X.columns[i]].apply(func=lambda x: 1 if x == self._modes[i] else -1) * self._importance[i]
            else:
                if pred_y is None:
                    pred_y = X[X.columns[i]].apply(func=lambda x: 1 if not pd.isna(x) and x <= self._modes[i] else -1) * \
                              self._importance[i]
                else:
                    pred_y += X[X.columns[i]].apply(func=lambda x: 1 if not pd.isna(x) and x <= self._modes[i] else -1) * self._importance[i]

        return np.array([self._labels[0] if pred < 0 else self._labels[1] for pred in pred_y])


class DummyBlackBoxSciKit(DummyBlackBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return self.predict_dummy(X)


class DummyBlackBoxTorch(DummyBlackBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, X):
        # should this return the vector format of torch?
        return self.predict_dummy(X)


def run_barbe_test_suite(dataset):
    blackbox = DummyBlackBoxSciKit(mode="unbiased", label_type="numeric")
    blackbox.fit(dataset, None)
    print(blackbox.predict(dataset))
    # BARBE with 100 perturbations
    barbe_explainer = BARBE(training_data=dataset,
                            perturbation_type='normal',
                            n_perturbations=1000,
                            dev_scaling_factor=1,
                            input_sets_class=False,
                            verbose=True)
    explanation = barbe_explainer.explain(dataset.iloc[0:1], blackbox)
    print(dataset.columns)
    print(blackbox._importance)
    print(explanation)
    counterfactuals = barbe_explainer.get_counterfactuals(dataset.iloc[0:1],
                                                          blackbox,
                                                          1 if blackbox.predict(dataset.iloc[0:1])[0] == -1 else -1,
                                                          n_counterfactuals=5)
    print(counterfactuals)
    # train BARBE with stats
    # get predictions from BARBE
    # get fidelity from BARBE
    # get counterfactuals from BARBE
    return

# ************************ BARBE TESTS ************************ #
# load data (mix of categorical and numerical)
cat_num_dataset = pd.read_csv("../barbe/dataset/weatherAUS.csv", index_col=None)
cat_num_dataset.drop(['Date', 'Location'], axis=1, inplace=True)
run_barbe_test_suite(cat_num_dataset)
# load data (only numerical)

# load data (only categorical)

# load data (repeat mix in numpy format)
