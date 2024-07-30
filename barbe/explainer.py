"""
This file contains the explainer part of BARBE and will call to all other parts of BARBE that make
 sense.
"""

import pickle
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


class BARBE:
    __doc__ = '''
        Purpose: **B**lack-box **A**ssocaition **R**ule-**B**ased **E**xplanation (**BARBE**) is a model-independent 
         framework that can explain the decisions of any black-box classifier for tabular datasets with high-precision. 

        Input: training_data (pandas DataFrame)   -> Sample of data used by a black box model to learn scale and some
                                                      feature interactions on.
               feature_names (list<string>)       -> feature names of input data if ungiven.
                | Default: None
               input_scale (list<float>)          -> scales of expected data if not given as training.
                | Default: None
               input_categories (dict<list>) or   -> indicator for which values are categorical and the possible values.
                |                (dict<dict>)         or direct assignment of values to labels.
                | Default: None
               verbose (boolearn)                 -> Whether to provide verbose output during training.
               mode (string)                      -> Mode BARBE will run in, important for generating simulated data.
                | Default: 'tabular' Options: {'tabular', 'text'}
               n_perturbations (int>1)            -> Number of perturbed sample points BARBE will generate to explain
                |                                     input data during BARBE.explain().
                | Default: 5000
               input_sets_class (boolean)         -> Whether the input data of BARBE.explain() sets the class label to
                |                                     True and False for training the SigDirect surrogate model.
                | Default: True
               dev_scaling_factor (None or int>0) -> Whether to reduce the deviation for perturbed data. Tends to 
                |                                     improve the surrogate performance as less is changed about the
                |                                     input data.
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
        # IAIN include surrogate model settings, make the model here?
        # IAIN do we include the perturbation type here? Is it consistent?
        # IAIN eventually method will include 'text' too.
        # IAIN raise warning when perturbations are not the default and text is called
        self._check_input_combination(training_data, feature_names, input_scale, input_categories)
        # IAIN renumber input categories if they are named instead

        # IAIN should include some checks in here for mode and perturbation validity

        self._dev_scaling = dev_scaling_factor is not None
        self._dev_scaling_factor = dev_scaling_factor if self._dev_scaling else 1
        self._verbose = verbose
        self._verbose_header = "BARBE:"
        self._n_perturbations = n_perturbations
        self._mode = mode
        self._feature_names = list(training_data) if feature_names is None else feature_names
        # self._perturber = LimeWrapper(training_data, training_labels)  # For our own perturber we only need to change this line
        # IAIN NOTES: adding a scaling factor to the deviation makes it work better
        # IAIN NOTES: normal distribution is more consistent for rules
        self._perturber = BarbePerturber(training_data=training_data,
                                         input_scale=input_scale,
                                         input_categories=input_categories,
                                         perturbation_type=perturbation_type,
                                         dev_scaling_factor=self._dev_scaling_factor,
                                         uniform_training_range=False,
                                         df=(None if training_data is not None else 50))
        self._perturbed_data = None
        self._input_sets_class = input_sets_class
        self._input_data = None
        self._blackbox_classification = {'input': None, 'perturbed': None}
        self._surrogate_classification = {'input': None, 'perturbed': None}
        self._surrogate_model = None  # SigDirect model trained on the new points
        self._explanation = "No explanations done yet."

    def __str__(self):
        return self._explanation

    def _check_input_combination(self, training_data, feature_names, input_scale, input_categories):
        # IAIN check that either training data is not None or all of feature names, scale, and categories have a value
        #  of a valid type. features names length must be the same as scale. Everything else is checked by perturber.
        pass

    def _fit_surrogate_model(self, input_data, input_model):
        # IAIN tabular and text perturbations differ in an important way
        if self._mode in 'tabular':
            self._generate_perturbed_tabular(input_data)
        elif self._mode in 'text':
            # IAIN should currently pass an error
            pass

        # wrap model so prediction call is always the same
        input_model = BlackBoxWrapper(input_model)

        if self._input_sets_class:  # black box is true or false for input class
            # IAIN need to clean up how data is passed to models (should be model side)
            input_model.set_class(input_model.predict(input_data.to_numpy().reshape(1,-1)))
        # get black box predictions
        self._blackbox_classification['input'] = input_model.predict(input_data.to_numpy().reshape(1,-1))  # IAIN make this look better
        self._blackbox_classification['perturbed'] = input_model.predict(self._perturbed_data)

        self._surrogate_model = SigDirectWrapper(self._feature_names, verbose=self._verbose)

        # fit the surrogate through the wrapper
        if self._verbose:
            print(self._verbose_header, 'black box prediction', self._blackbox_classification['perturbed'])
        self._surrogate_model.fit(self._perturbed_data, self._blackbox_classification['perturbed'])

        self._surrogate_classification['input'] = self._surrogate_model.predict(input_data.to_numpy().reshape(1,-1))
        self._surrogate_classification['perturbed'] = self._surrogate_model.predict(self._perturbed_data)
        if self._verbose:
            print(self._verbose_header, "was it successful?", self._blackbox_classification['input'], self._surrogate_classification['input'])
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
        return self._surrogate_model.get_contrast_sets(data_row)

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
        Input: None
        Purpose: Get all the rules from the surrogate model.
        Output: list<(rule_text, support, confidence)>, all rules with their support and confidence.
        """
        return self._surrogate_model.get_all_rules()

    def get_perturber(self, feature='all'):
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

    def explain(self, input_data, input_model, fit_new_surrogate=True):
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
            # IAIN push a warning later
            fit_new_surrogate = True

        if fit_new_surrogate:
            # fit the SigDirect model
            fit_tries = 0
            fit_success = False
            while fit_tries < 5 and not fit_success:
                fit_tries += 1
                fit_success = self._fit_surrogate_model(input_data, input_model)
            if not fit_success:
                raise Exception('BARBE ERROR: model did not successfully match input data in 5 tries.')
            if self._verbose:
                print(self._verbose_header, 'number of tries:', fit_tries, fit_success)

        if self._verbose:
            print(self._verbose_header, 'fidelity:', self.get_surrogate_fidelity())
        # expecting a fit model explain the result for the new data, unless refit it assumes the prediction is correct
        # explanation_temp = self.get_features(input_data, self._blackbox_classification['input'][0])
        self._explanation = (str(self.get_features(input_data,
                                                   self._surrogate_model.predict(input_data.to_numpy().reshape(1, -1))[0]))
                             + " \n " + str(self.get_surrogate_fidelity()))

        return self._explanation
