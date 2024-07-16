"""
Perturbs input data based on different modes.

This function should have the same functionality that the LimeWrapper in lime_interface.py uses.
"""

import numpy as np
from numpy.random import Generator, PCG64


class BarbePerturber:
    # TODO: current version only works for numerical data ? How do we handle and detect discrete data
    #  TODO: consider assigning numbers to the categories and on return we select category based on the closes value
    def __init__(self, training_data, perturbation_type='uniform', covariance_mode='full', rng_distribution='uniform',
                 training_uniform=False, dev_scaling_factor=1, df=None):
        self._covariance_mode = covariance_mode
        self._perturbation_type = perturbation_type
        self._n_features = training_data.shape[1]
        self._df = training_data.shape[0] if df is None else df  # note if over 100 it is essentially normal
        self._means = self._calculate_means(training_data.to_numpy())  # required to recenter input
        self._scale = self._calculate_scale(training_data.to_numpy()) / dev_scaling_factor  # required to scale
        self._training_uniform = training_uniform
        self._max, self._min = self._calculate_range(training_data.to_numpy())
        self._covariance = self._calculate_covariance(training_data.to_numpy()) / dev_scaling_factor

        self._distribution = rng_distribution
        self._random_state = Generator(PCG64())

    def _calculate_range(self, training_array):
        if self._training_uniform:
            return np.max(training_array, axis=1), np.min(training_array, axis=1)
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
            if self._training_uniform:
                return self._random_state.uniform(self._min, self._max, size=(num_perturbations, self._n_features))
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

    def produce_perturbation(self, num_perturbations, data_row=None):
        if data_row is None:
            data_row = self._means
        # returns perturbed data
        perturbed_data = self._fetch_perturbation_rows(data_row.to_numpy().reshape(1, -1), num_perturbations)
        perturbed_data[0, :] = data_row.to_numpy()  # make sure to include the row in training
        return perturbed_data
