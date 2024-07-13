"""
Perturbs input data based on different modes.

This function should have the same functionality that the LimeWrapper in lime_interface.py uses.
"""


class BarbePerturber:
    def __init__(self, training_data, perturbation_type='normal'):
        self._perturbation_type = perturbation_type
        self._scale = []  # required to scale back up input
        self._means = []  # required to recenter input
        self._covariance = [[]]  # required to generate accurate perturbations

    def _calculate_scale(self, training_data):
        pass

    def _calculate_means(self, training_data):
        pass

    def _calculate_covariance(self, training_data):
        pass

    def produce_perturbations(self, data_row, num_perturbations):
        pass
