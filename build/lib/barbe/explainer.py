"""
This file contains the explainer part of BARBE and will call to all other parts of BARBE that make
 sense.
"""

from utils.sigdirect_interface import SigDirectWrapper
import pickle
import pandas as pd


class BARBE:
    def __init__(self, verbose=False, mode='tabular', n_perturbations=1000):
        # IAIN include model settings, make the model here?
        # IAIN do we include the perturbation type here? Is it consistent?
        # IAIN eventually method will include 'text' too.
        # IAIN raise warning when perturbations are not the default and text is called
        self._verbose = verbose
        self._n_perturbations = n_perturbations
        self._mode = mode
        self._perturbed_data = None
        self._blackbox_classification = {'input': None, 'perturbed': None}
        self._surrogate_model = None  # SigDirect model trained on the new point

    def _fit_surrogate_model(self):
        # IAIN tabular and text perturbations differ in an important way
        pass

    def _check_input_data(self, input_data):
        # IAIN checks if input is valid (tabular data is tabular, text data is text)
        #  and put it into a common format (pandas dataframe or string)
        pass

    def _check_input_model(self, input_model):
        # IAIN ensures that the input model has a predict function and output is as expected
        #  or can be formatted into a list (,k)
        pass

    def _generate_perturbed_tabular(self, input_data):
        # calls to set method to perturb the data, based on the mode
        pass

    def _generate_perturbed_text(self, input_data):
        # calls to set method to perturb the data, based on the mode
        pass

    def get_surrogate_model(self):
        # IAIN should deep copy or go into utilities
        return self._surrogate_model

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=None):
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        pass

    def get_rules(self):
        # IAIN this will output rules and their translations as learned by the model
        pass

    def get_features(self):
        # IAIN same as previous named get features
        pass

    def explain(self, input_data, input_model, fit_surrogate=True):
        # IAIN should include the surrogate similarity to original model
        # IAIN follow how it is called at the moment
        # IAIN the way it is called by them kinda sucks
        pass
