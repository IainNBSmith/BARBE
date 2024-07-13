"""

"""
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances
from numpy.random import RandomState


class LimeWrapper:
    def __init__(self, training_data, class_label):
        # IAIN should check that this is dataframe, this is only to fix class
        self._column_names = list(training_data)
        ordered_class_labels = class_label
        training_data = training_data
        all_features = training_data.columns.values
        columns = list(training_data.columns)
        categorical_features = [x for x in columns if '_' in str(x)]
        categorical_feature_indices = [columns.index(x) for x in columns if '_' in str(x)]
        categorical_features_map = {columns.index(x): x for x in columns if '_' in str(x)}
        self.lt = LimeTabularExplainer(training_data.values,
                                       categorical_features=categorical_feature_indices,
                                       feature_names=all_features,
                                       verbose=False,
                                       class_names=ordered_class_labels,
                                       mode='classification',
                                       sample_around_instance=True,
                                       random_state=RandomState(1),
                                       discretizer='decile')

    def _make_pandas_dataframe(self, data):
        return pd.DataFrame(data=data, columns=self._column_names)

    def produce_perturbation(self, data_row, num_samples):
        data, inverse =  self.lt._LimeTabularExplainer__data_inverse(data_row, num_samples)
        data[0, :] = data_row.to_numpy()
        inverse[0, :] = data_row.to_numpy()
        return data, inverse
