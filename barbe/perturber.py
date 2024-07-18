"""
Perturbs input data based on different modes.

This function should have the same functionality that the LimeWrapper in lime_interface.py uses.
"""

import numpy as np
from numpy.random import Generator, PCG64


class BarbePerturber:
    def __init__(self, training_data, perturbation_type='uniform', covariance_mode='full', rng_distribution='uniform',
                 uniform_training_range=False, uniform_scaled=True, dev_scaling_factor=1, df=None):
        # check and modify training data
        self._categorical_features = []
        self._feature_original_types = []
        self._categorical_key = dict()

        training_data = self._training_discrete_conversion(training_data.to_numpy())

        self._covariance_mode = covariance_mode
        self._perturbation_type = perturbation_type
        self._n_features = training_data.shape[1]
        self._df = training_data.shape[0] if df is None else df  # note if over 100 it is essentially normal
        self._means = self._calculate_means(training_data)  # required to recenter input
        self._scale = self._calculate_scale(training_data) / dev_scaling_factor  # required to scale
        self._uniform_training_range = uniform_training_range  # whether uniform data should be generated based on the training data range
        self._uniform_scaled = uniform_scaled  # whether uniform perturbation should scale to deviation in training data (independent from training range)
        self._max, self._min = self._calculate_range(training_data)
        self._covariance = self._calculate_covariance(training_data) / dev_scaling_factor

        self._distribution = rng_distribution
        self._random_state = Generator(PCG64())

    def _training_discrete_conversion(self, training_array, category_threshold=10):
        # conversion from discrete values -> numeric values
        for i in range(training_array.shape[1]):
            unique_values = np.unique(training_array[:, i])
            if len(unique_values) <= category_threshold:
                self._categorical_features.append(i)
                self._feature_original_types.append(type(unique_values[0]))
                self._categorical_key[i] = dict()
                for j in range(len(unique_values)):
                    value = str(unique_values[j])
                    self._categorical_key[i][value] = j
                    training_array[((training_array[:, i]).astype(str) == value), i] = j

        return training_array.astype(float)

    def _conversion_input(self, input_array):
        for i in self._categorical_features:
            input_array[i] = self._categorical_key[i][str(input_array[i])]
        return input_array

    def _perturbed_discrete_conversion(self, perturbed_array):
        # conversion from numeric values -> discrete values
        def nearest_values(x, y):  # utility in one function for finding the nearest values in a list
            y = np.array(y)
            near_array = []
            for val in x:
                pot_vals = np.abs(y - val)
                ind_min = np.argmin(pot_vals)
                near_array.append(ind_min)
            return y[near_array]

        perturbed_array = perturbed_array.astype(object)
        for i in self._categorical_features:
            perturbed_array[:, i] = nearest_values(perturbed_array[:, i],
                                                   [item for key, item in self._categorical_key[i].items()])
            replacement_values = np.array([None for i in range(perturbed_array.shape[0])])
            for dvalue in self._categorical_key[i].keys():
                replacement_values[(perturbed_array[:, i] == self._categorical_key[i][str(dvalue)])] = dvalue
            perturbed_array[:, i] = replacement_values.astype(self._feature_original_types[i])
        return perturbed_array

    def _calculate_range(self, training_array):
        if self._uniform_training_range:
            return np.max(training_array, axis=0), np.min(training_array, axis=0)
        # Use 0 to 1 instead and scale result
        return (np.array([2 for _ in range(training_array.shape[1])]),
                np.array([-2 for _ in range(training_array.shape[1])]))

    def _calculate_scale(self, training_array):
        return np.std(training_array, axis=0)

    def _calculate_means(self, training_array):
        return np.mean(training_array, axis=0)

    def _rescale_data(self, unscaled_data, scaling_mean=None):
        if scaling_mean is None:
            scaling_mean = self._means
        return (unscaled_data * self._scale) + scaling_mean

    def _calculate_covariance(self, training_array):
        full_cov = np.cov(training_array.T)
        if self._covariance_mode in 'full':
            return full_cov
        elif self._covariance_mode in 'diagonal':
            return np.diag(np.diag(full_cov))
        return None

    def _fetch_perturbation_rows(self, row_array, num_perturbations):
        if self._distribution in 'uniform':
            # low high size
            if self._uniform_training_range:
                return self._random_state.uniform(self._min, self._max, size=(num_perturbations, self._n_features))

            if not self._uniform_scaled:
                return (self._random_state.uniform(self._min, self._max, size=(num_perturbations, self._n_features)) +
                        row_array)

            return self._rescale_data(self._random_state.uniform(self._min, self._max, size=(num_perturbations,
                                                                                             self._n_features)),
                                      scaling_mean=row_array)
        elif self._distribution in 'normal':
            # location scale size
            return self._random_state.multivariate_normal(row_array.flatten(), self._covariance, size=num_perturbations)
        elif self._distribution in 'cauchy':
            # size (requires scaling)
            return self._rescale_data(self._random_state.standard_cauchy(size=(num_perturbations, self._n_features)),
                                      scaling_mean=row_array)
        elif self._distribution in 't-distribution':
            # df size
            return self._rescale_data(self._random_state.standard_t(self._df, size=(num_perturbations, self._n_features)),
                                      scaling_mean=row_array)
        return None

    def get_discrete_values(self):
        return self._categorical_key.copy()

    def produce_perturbation(self, num_perturbations, data_row=None):
        if data_row is None:
            data_row = self._means
        else:
            data_row = self._conversion_input(data_row.to_numpy())
        # returns perturbed data
        perturbed_data = self._fetch_perturbation_rows(data_row, num_perturbations)
        perturbed_data[0, :] = data_row  # make sure to include the row in training
        perturbed_data = self._perturbed_discrete_conversion(perturbed_data)
        return perturbed_data
