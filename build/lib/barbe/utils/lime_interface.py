"""

"""
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances
from numpy.random import RandomState


class LimeWrapper:
    def __init__(self, training_data):
        # IAIN should check that this is dataframe, this is only to fix class
        self._column_names = list(training_data)[:-1]
        ordered_class_labels = sorted(list(set(training_data['class'].values)))
        training_data = training_data.drop(['class'], axis=1)
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

    def produce_perturbation(self, data_row, num_samples, bbmodel):
        # TODO
        # IAIN when the encoder is written into sigdirect then we need only call the
        #  lime method for inverse rather than worry about this one

        # IAIN may need to scale the data inside of this function
        # self.__data_inverse_barbe(data_row, num_samples, predict_fn, barbe_mode)
        data, inverse, sd_data, ohe = self.lt._LimeTabularExplainer__data_inverse_barbe(data_row.values, num_samples,
                                                                                        bbmodel, "BARBE")
        scaled_data = (data - self.lt.scaler.mean_) / self.lt.scaler.scale_
        print("EARLIER IAIN")
        print(ohe)
        return self._make_pandas_dataframe(data), inverse, sd_data, ohe
