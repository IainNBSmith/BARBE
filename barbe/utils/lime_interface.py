"""

"""
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances


class LimeWrapper:
    def __init__(self, training_data):
        self.lt = LimeTabularExplainer(training_data)

    def produce_perturbation(self, data_row, num_samples):
        # IAIN may need to scale the data inside of this function
        data, inverse = self.lt.__data_inverse(data_row, num_samples)
        scaled_data = (data - self.lt.scaler.mean_) / self.lt.scaler.scale_
        return data, inverse, scaled_data, None
