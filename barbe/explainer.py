"""
This file contains the explainer part of BARBE and will call to all other parts of BARBE that make
 sense.
"""

import pickle
import warnings

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from copy import deepcopy
import numpy as np

# Import wrappers for other packages
from barbe.utils.sigdirect_interface import SigDirectWrapper
# from barbe.utils.lime_interface import LimeWrapper
from barbe.utils.bbmodel_interface import BlackBoxWrapper
from barbe.perturber import BarbePerturber
from barbe.counterfactual import BarbeCounterfactual


DIST_INFO = {'Generic Distributions': {'uniform': 'Uniform', 'normal': 'Normal'},
             'Skewed Distributions': {'t-distribution': 't-Distribution', 'cauchy': 'Cauchy'}}


class BARBE:
    __doc__ = '''
        Purpose: **B**lack-box **A**ssociation **R**ule-**B**ased **E**xplanation (**BARBE**) is a model-independent 
         framework that can explain the decisions of any black-box classifier for tabular datasets with high-precision. 

        Input: training_data (pandas DataFrame)   -> Sample of data used by a black box model to learn scale and some
                |                                     feature interactions on.
                | Default: None
               feature_names (list<string>)       -> feature names of input data if ungiven.
                | Default: None
               input_scale (list<float>)          -> scales of expected data if not given as training.
                | Default: None
               input_categories (dict<list>) or   -> indicator for which values are categorical and the possible values.
                |                (dict<dict>)         or direct assignment of values to labels.
                | Default: None
               verbose (boolean)                  -> Whether to provide verbose output during training.
               mode (string)                      -> Mode BARBE will run in, important for generating simulated data.
                | Default: 'tabular' Options: {'tabular', 'text'}
               n_perturbations (int>1)            -> Number of perturbed sample points BARBE will generate to explain
                |                                     input data during BARBE.explain().
                | Default: 5000
               input_sets_class (boolean)         -> Whether the input data of BARBE.explain() sets the class label to
                |                                     True and False for training the SigDirect surrogate model.
                | Default: True
               perturbation_type (string)         -> The type of distribution to use when generating perturbed data.
                |                                     As referred to in BarbePurturber.
                |                                     'uniform' -> uniform distribution over a range (-2, 2) is equally 
                |                                                   as likely to generate 0 as 0.1 as 1.2
                |                                     'normal' -> normal distribution, data will be more similar to the
                |                                                  true distribution of the data along with all
                |                                                  interactions between features.
                |                                     'cauchy' -> cauchy distribution, long-tailed distribution that
                |                                                  captures more radical differences in feature values.
                |                                                  Useful when edge cases are a concern.
                |                                     't-distribution' -> t distribution, useful when less training data
                |                                                          is available (for example if privacy is a 
                |                                                          concern). Has wider tails and more 
                |                                                          flexibility than the normal distribution.
                |                                                          Set to 20 df if training not given.
                | Default: 'uniform' Options: {'uniform', 'normal', 'cauchy', 't-distribution'}
               dev_scaling_factor (None or int>0) -> Whether to reduce the deviation for perturbed data. Tends to 
                |                                     improve the surrogate performance as less is changed about the
                |                                     input data. IGNORED if scales are given.
                | Default: 5
    
        Example Usage:
        from sklearn import datasets
        from sklearn.ensemble import RandomForestClassifier()
        import pandas as pd
        
        iris = datasets.load_iris()

        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

        data_row = iris_df.iloc[50]
        training_labels = iris.target
        training_data = iris_df

        print("Running test: BARBE iris Run")
        start_time = datetime.now()
        bbmodel = RandomForestClassifier()
        bbmodel.fit(training_data, training_labels)

        explainer = BARBE(training_data=training_data, verbose=True, input_sets_class=True)
        explanation = explainer.explain(data_row, bbmodel)
        print("Test Time: ", datetime.now() - start_time)
        print(data_row)
        print(explanation)
        print(bbmodel.feature_importances_)
        print("ALL RULES:", explainer.get_rules())
        
        '''

    def __init__(self, training_data=None, feature_names=None, input_scale=None, input_categories=None,
                 verbose=False, mode='tabular', input_sets_class=True,
                 n_perturbations=5000, perturbation_type='uniform', dev_scaling_factor=5):

        self._check_input_combination(training_data, feature_names, input_scale, input_categories, n_perturbations,
                                      dev_scaling_factor)
        input_categories = self._fix_input_categories(input_categories, feature_names)

        self._dev_scaling = dev_scaling_factor is not None
        self._dev_scaling_factor = dev_scaling_factor \
            if self._dev_scaling and not input_scale else 1
        self._verbose = verbose
        self._verbose_header = "BARBE:"
        self._n_perturbations = n_perturbations
        self._mode = mode
        self._feature_names = list(training_data) \
            if feature_names is None else feature_names
        # OLD CODE WHEN LIME PERTURBER WAS USED
        # self._perturber = LimeWrapper(training_data, training_labels)
        self._perturber = BarbePerturber(training_data=training_data,
                                         input_scale=input_scale,
                                         input_categories=input_categories,
                                         perturbation_type=perturbation_type,
                                         dev_scaling_factor=self._dev_scaling_factor,
                                         uniform_training_range=False,
                                         df=(None if training_data is not None else 20))
        self._perturbed_data = None
        self._input_sets_class = input_sets_class
        self._input_data = None
        self._blackbox_classification = {'input': None, 'perturbed': None}
        self._surrogate_classification = {'input': None, 'perturbed': None}
        self._surrogate_model = None  # SigDirect model trained on the new points
        self._counterfactual = BarbeCounterfactual()
        self._explanation = "No explanations done yet."

    def __str__(self):
        return self._explanation

    def _check_input_combination(self, training_data, feature_names, input_scale, input_categories, n_perturbations,
                                 dev_scaling_factor):
        error_header = "BARBE Check Input"
        # check that there is either training data or an input scale
        if training_data is None and input_scale is None:
            assert ValueError(error_header + " either training data must be provided or input scales must be given.")
        if training_data is not None and input_scale is not None:
            assert ValueError(error_header + " both training data and scale cannot be given.")

        if input_scale is not None:
            # if there is an input scale make sure feature names are given
            if len(input_scale) != len(feature_names):
                assert ValueError(error_header + " must have a feature name for each scale value.")
            # make sure that any input categories have keys described in scale
            if list(input_categories.keys()) not in feature_names and \
                    list(input_categories.keys()) not in range(len(input_scale)):
                assert ValueError(error_header + " all scales must be set for features, even categorical. Size of "
                                                 "scales should be set relative to the number of categories.")

        # check that number of perturbations is appropriate
        if n_perturbations < 100:
            warnings.warn(error_header + " low number of perturbations (" + str(n_perturbations) + ") may lead to "
                                         "misrepresentation of the underlying model.")

    def _fix_input_categories(self, input_categories, feature_names):
        if input_categories is None:
            return dict()
        new_input_categories = dict()
        # make input categories numerical rather than named if not named
        for key in input_categories.keys():
            temp_key = feature_names.index(key) if key in feature_names else int(key)
            new_input_categories[temp_key] = input_categories[key]
        return new_input_categories

    def _fit_surrogate_model(self, input_data, input_model):
        # differently produce perturbations for text and tabular data
        if self._mode in 'tabular':
            self._generate_perturbed_tabular(input_data)
        elif self._mode in 'text':
            self._generate_perturbed_text(input_data)

        # wrap model so prediction call is always the same
        input_model = BlackBoxWrapper(input_model)

        if self._input_sets_class:  # black box is true or false for input class
            input_model.set_class(input_model.predict(input_data.to_numpy().reshape(1,-1)))

        # get black box predictions
        self._blackbox_classification['input'] = input_model.predict(input_data.to_numpy().reshape(1, -1))  # IAIN make this look better
        self._blackbox_classification['perturbed'] = input_model.predict(self._perturbed_data)

        # fit and get predictions for surrogate model
        self._surrogate_model = SigDirectWrapper(self._feature_names, verbose=self._verbose)
        self._surrogate_model.fit(self._perturbed_data, self._blackbox_classification['perturbed'])

        self._surrogate_classification['input'] = self._surrogate_model.predict(input_data.to_numpy().reshape(1, -1))
        self._surrogate_classification['perturbed'] = self._surrogate_model.predict(self._perturbed_data)

        if self._verbose:
            print(self._verbose_header, "was it successful?", self._blackbox_classification['input'],
                  self._surrogate_classification['input'])

        return self._blackbox_classification['input'] == self._surrogate_classification['input']

    def _check_input_data(self, input_data):
        # IAIN checks if input is valid (tabular data is tabular, text data is text)
        #  and put it into a common format (pandas dataframe or string)
        if self._mode in 'tabular':
            return input_data
        elif self._mode in 'text':
            # IAIN should currently pass an error
            pass
        pass

    def _check_input_model(self, input_model):
        # IAIN ensures that the input model has a predict function and output is as expected
        #  or can be formatted into a list (,k)
        if self._mode in 'tabular':
            try:
                bbmodel = BlackBoxWrapper(input_model)
            except:
                raise Exception('Error while checking black box model.')
        elif self._mode in 'text':
            # IAIN should currently pass an error
            pass
        pass

    def _generate_perturbed_tabular(self, input_data):
        # calls to set method to perturb the data, based on the mode
        self._perturbed_data = self._perturber.produce_perturbation(self._n_perturbations,
                                                                    data_row=input_data)

    def _generate_perturbed_text(self, input_data):
        # calls to set method to perturb the data, based on the mode
        # IAIN pass a warning / error
        self._perturbed_data = None
        pass

    def get_surrogate_model(self):
        return deepcopy(self._surrogate_model)

    def get_contrasting_rules(self, data_row):
        """
        Input: data_row (pandas DataFrame row) -> row to find contrast rules for
        Purpose: Find rules that apply directly to a row's values that may decide classification results in surrogate.
                  For use in counterfactual reasoning.
        Output: list<(string, int, float, float, float)> -> list of contrast rules, their class, p-value, and importance
        """
        return self._surrogate_model.get_contrast_sets(data_row)

    def get_counterfactual_explanation(self, data_row):
        """
        Input:
        Purpose:
        Output:
        """
        data_cls = self._surrogate_model.predict(data_row.to_numpy().reshape(1, -1))[0]
        print("IAIN getting rules + ohe simple")
        aa = self._surrogate_model.get_contrast_sets(data_row, raw_rules=True, max_dev=0.05)
        print("IAIN CONTRAST ", aa)
        self._counterfactual.fit(self._surrogate_model.get_contrast_sets(data_row, raw_rules=True, max_dev=0.00005),
                                 self._surrogate_model.get_ohe_simple())
        print("IAIN getting prediction")
        counter_predict, counter_rules = self._counterfactual.predict(self._surrogate_model._encode(data_row.to_numpy().reshape(1, -1)), data_cls)
        print(counter_predict)
        # counter_predict = self._surrogate_model._decode([counter_predict])
        new_class = self._surrogate_model._sigdirect_model.predict(counter_predict)
        counter_predict = self._surrogate_model._decode([counter_predict])
        return counter_predict, counter_rules, new_class


    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score):
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        return comparison_method(self._blackbox_classification['perturbed'],
                                 self._surrogate_classification['perturbed'])

    def get_rules(self):
        """
        Purpose: Get all the rules from the surrogate model.
        Output: list<(rule_text, support, confidence)>, all rules with their support and confidence.
        """
        return self._surrogate_model.get_all_rules()

    def get_perturbed_data(self):
        return self._perturbed_data.copy()

    def get_perturber(self, feature='all'):
        """
        Input: feature (string) -> String indicating the type of feature to make a 'get' call to the  purtuber for.
                | Default: 'all' Options: {'all', 'scale', 'categories'}
        Purpose: Get perturber or information from it.
        Output: instance of BarbePertuber or list<object>.
        """
        if feature in 'all':
            return self._perturber
        elif feature in 'scale':
            return self._perturber.get_scale()
        elif feature in 'categories':
            return self._perturber.get_discrete_values()
        return None

    def get_features(self, input_data, true_label):
        """
        Input: input_data (pandas DataFrame row)    -> Input to get feature usage from.
               true_label (string, int, or boolean) -> True label that would be assigned by the black box model. Must be
                |                                       set to a boolean if using input_sets_class = True.
        Purpose: Get the features and their cumulative support from the surrogate model, negative values indicate a
                  rule is present but not used when predicting the input data.
        Output: list<(feature, sum of support)>, all features used when predicting the input data and the total support.
        """
        return self._surrogate_model.get_features(input_data, true_label)

    def get_categories(self):
        """
        Input: None.
        Purpose: Get the features that are defined as categorical by the surrogate model.
        Output: dict<(feature column number, list<obj> -> len<=10)>
        """
        self._surrogate_model.get_categories()

    def explain(self, input_data, input_model, fit_new_surrogate=True, ignore_errors=False):
        """
        Input: input_data (pandas DataFrame row)                 -> The data to explain.
               input_model (callable or obj with predict method) -> Input model to provide explanation over.
               fit_new_surrogate (boolean)                       -> Whether to fit a new surrogate model to explain the
                |                                                    input data. Does not use input_model.
        Purpose: Provide the importance of different features for classifying the input data as used by the input model.
        Output: explanation, a string consisting of feature importance and surrogate fidelity (how similar the
                 predictions made by BARBE are to the black box model).
        """
        # check input before going further
        self._check_input_model(input_model)
        input_data = self._check_input_data(input_data)

        # cannot test without a new surrogate
        if self._surrogate_model is None:
            fit_new_surrogate = True

        if fit_new_surrogate:
            # fit the SigDirect model
            fit_tries = 0
            fit_success = False
            while fit_tries < 5 and not fit_success:
                fit_tries += 1
                fit_success = self._fit_surrogate_model(input_data, input_model)
            if not fit_success:
                if not ignore_errors:
                    raise Exception('BARBE ERROR: model did not successfully match input data in 5 tries.')
                else:
                    return None
            if self._verbose:
                print(self._verbose_header, 'number of tries:', fit_tries, fit_success)

        if self._verbose:
            print(self._verbose_header, 'fidelity:', self.get_surrogate_fidelity())
        # expecting a fit model explain the result for the new data, unless refit it assumes the prediction is correct
        # explanation_temp = self.get_features(input_data, self._blackbox_classification['input'][0])
        self._explanation = self.get_features(input_data, self._surrogate_model.predict(input_data.to_numpy().reshape(1, -1))[0])

        return self._explanation
