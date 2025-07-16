"""
This file contains the explainer part of BARBE and will call to all other parts of BARBE that make
 sense.
"""

# TODO: add encoding named predictions

import pickle
import warnings
from collections import Counter

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from barbe.utils.evaluation_measures import euclidean_weights, nearest_neighbor_weights

from copy import deepcopy
import numpy as np
import math

# Import wrappers for other packages
from barbe.utils.sigdirect_interface import SigDirectWrapper
# from barbe.utils.lime_interface import LimeWrapper
from barbe.utils.bbmodel_interface import BlackBoxWrapper
from barbe.perturber import BarbePerturber, ClassBalancedPerturber
from barbe.counterfactual import BarbeCounterfactual


DIST_INFO = {'Generic Distributions': {'uniform': 'Uniform', 'normal': 'Normal'},
             'Skewed Distributions': {'t-distribution': 't-Distribution', 'cauchy': 'Cauchy (*extreme cases)'}}
BOUND_INFO = {'Loop-Based Bounding': {'--': 'Cut to Mean', '---': 'Cut to Sample'},
              'Absolute Bounding': {'----': 'Set to Remainder (Max-Min)', '-----': 'Set to Max/Min'}}


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
               input_bounds (list<(floats)>)      -> list of bounds in order (min, max), (None, max), (min, None), or None
                |                                     setting bounds for each variable in the final output. Will be
                |                                     checked for values that will easily be exceeded.
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
               n_bins (int>0)                     -> Number of bins to use on continuous numerical data.
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
    # ************** GLOBALLY USED SETTINGS ************** #
    DEFAULT_STANDARDIZED_CATEGORY_VARIANCE = 0.3334  # in perturber categories will have a consistent variance
    # DEFAULT_STANDARDIZED_CATEGORY_VARIANCE = None  # run perturber with variances the same as before
    DEFAULT_T_DIST_FREEDOM = 20  # when using t-distribution with no data sets the degrees of freedom

    # ************** UNIMPLEMENTED SETTINGS ************** #
    DEFAULT_DEVIATION_SCALING_METHODS = {  # reason: requires testing to implement, change dev scaling to call or int
        'normal': lambda x: x**2,
        'uniform': lambda x: x/2,
        't-distribution': lambda x: (x**2)/3,
        'cauchy': lambda x: x/10
    }  # handles modification to deviation that improve performance
    DEFAULT_CATEGORY_FREQUENCY_ODDS = True  # reason: requires testing, need to remove the option if this is better

    def __init__(self, training_data=None,
                 feature_names=None, input_scale=None, input_categories=None, input_covariance=None, input_means=None,
                 input_bounds=None, input_sets_class=True,
                 mode='tabular',  n_perturbations=5000, n_bins=5, perturbation_type='uniform', dev_scaling_factor=5,
                 higher_frequent_category_odds=True,
                 balance_classes=True,
                 learn_negation_rules=True,
                 verbose=False):

        # IAIN consider adding more options to bound setting for BARBE side
        # TODO: add more requirement checks
        #  like: - higher_frequent_category_odds=True requires that input_means is given (at least for the categorical
        #           values) - we may want to modify format of the means passed into the method to make usage clearer
        self._check_input_combination(training_data, feature_names, input_scale, input_categories, n_perturbations,
                                      dev_scaling_factor)
        #input_categories = self._fix_input_categories(input_categories, feature_names)

        self._learn_negation_rules = learn_negation_rules
        self._dev_scaling = dev_scaling_factor is not None
        self._dev_scaling_factor = dev_scaling_factor \
            if self._dev_scaling and input_scale is None else 1
        self._verbose = verbose
        self._verbose_header = "BARBE:"
        self._n_perturbations = n_perturbations
        self._mode = mode
        self._balanced_classes = balance_classes
        self._feature_names = list(training_data) \
            if feature_names is None else feature_names
        # OLD CODE WHEN LIME PERTURBER WAS USED
        # self._perturber = LimeWrapper(training_data, training_labels)
        PertClass = BarbePerturber if not self._balanced_classes else ClassBalancedPerturber
        self._perturber = PertClass(training_data=training_data,
                                    input_scale=input_scale,
                                    input_categories=input_categories,
                                    input_bounds=input_bounds,
                                    input_covariance=input_covariance,
                                    input_means=input_means,
                                    input_feature_names=self._feature_names,
                                    perturbation_type=perturbation_type,
                                    standardized_categorical_variance=BARBE.DEFAULT_STANDARDIZED_CATEGORY_VARIANCE,
                                    use_mean_categorical_odds=higher_frequent_category_odds,
                                    dev_scaling_factor=self._dev_scaling_factor,
                                    uniform_training_range=False,
                                    df=(None if training_data is not None else BARBE.DEFAULT_T_DIST_FREEDOM))
        self._perturbed_data = None
        self._input_sets_class = input_sets_class
        self._set_class = None
        self._input_data = None
        self._blackbox_classification = {'input': None, 'perturbed': None}
        self._surrogate_classification = {'input': None, 'perturbed': None}
        self._surrogate_model = None  # SigDirect model trained on the new points
        self._counterfactual = BarbeCounterfactual()
        self._n_bins = n_bins
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
        # wrap model so prediction call is always the same
        input_model = BlackBoxWrapper(input_model)

        input_row = pd.DataFrame(columns=self._feature_names, index=[0])
        input_row.loc[0, :] = input_data.to_numpy().reshape((1, -1))

        if self._input_sets_class:  # black box is true or false for input class
            self._set_class = input_model.predict(input_row.copy())
            input_model.set_class(self._set_class)

        if self._mode in 'tabular':
            self._generate_perturbed_tabular(input_data.copy(), bbmodel=input_model)
        elif self._mode in 'text':
            self._generate_perturbed_text(input_data)

        # get black box predictions
        self._blackbox_classification['input'] = input_model.predict(input_row.copy())  # IAIN make this look better
        #print(self._perturbed_data)
        self._blackbox_classification['perturbed'] = input_model.predict(pd.DataFrame(self._perturbed_data.copy(),
                                                                                      columns=self._feature_names))
        #print(input_data)
        #print(pd.DataFrame(self._perturbed_data, columns=self._feature_names).iloc[0])
        # fit and get predictions for surrogate model
        self._surrogate_model = SigDirectWrapper(self._feature_names, n_bins=self._n_bins,
                                                 use_negative_rules=self._learn_negation_rules,
                                                 verbose=self._verbose)
        self._surrogate_model.fit(self._perturbed_data.copy(), self._blackbox_classification['perturbed'])

        self._surrogate_classification['input'] = self._surrogate_model.predict(input_row.to_numpy().reshape((1, -1)).copy())
        self._surrogate_classification['perturbed'] = self._surrogate_model.predict(self._perturbed_data.copy())

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

    def _generate_perturbed_tabular(self, input_data, bbmodel=None):
        # calls to set method to perturb the data, based on the mode
        if not self._balanced_classes:
            # ignore model if not making balanced classes
            self._perturbed_data = self._perturber.produce_perturbation(self._n_perturbations,
                                                                        data_row=input_data)
        else:
            self._perturbed_data = self._perturber.produce_balanced_perturbation(self._n_perturbations,
                                                                                 bbmodel,
                                                                                 data_row=input_data)

    def _generate_perturbed_text(self, input_data):
        # calls to set method to perturb the data, based on the mode
        # IAIN pass a warning / error
        self._perturbed_data = None
        pass

    def get_available_classes(self):
        return np.unique(self._blackbox_classification['perturbed']).tolist()

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

    def get_counterfactuals(self, data_row, bbmodel, wanted_class, n_counterfactuals=1, importance_counterfactuals=False,
                            restricted_features=None, prioritize_importance=False):
        scale_list = self._perturber.get_scale()
        mean_list = self._perturber.get_means()
        scale_mean_list = [(scale_list[i], mean_list[i]) for i in range(len(scale_list))]
        #print(scale_mean_list)
        #assert False

        original_enc = self._surrogate_model._encode(data_row.to_numpy().reshape((1, -1)).copy())
        original_dec = self._surrogate_model._decode(original_enc)
        #print(original_enc)
        #print(original_dec)
        #print(data_row.to_numpy().reshape((1, -1)))

        self._counterfactual.fit(self,
                                 feat_scales=scale_mean_list,
                                 importance_counterfactuals=importance_counterfactuals,
                                 prioritize_importance=prioritize_importance)
        return self._counterfactual.predict(data_row, bbmodel, new_class=wanted_class, n_counterfactuals=n_counterfactuals,
                                            restricted_features=restricted_features)

    def get_counterfactual_explanation(self, data_row, wanted_class, n_counterfactuals=1):
        """
        Input: data_row (pandas Series)  -> data row that should be modified to change class.
               wanted_class (string)     -> string value of desired class from black box, found via
                                             get_available_classes()
               n_counterfactuals (int>0) -> number of counterfactual examples with slight differences to produce.
                | Default: 1
        Purpose: Produce a counterfactual explanation, a modification to existing data to produce a new class, based on
                  the rules learned by SigDirect.
        Output: counter_value (list)                          -> new values assigned to the original data.
                counter_rules (list<original_rule, new_rule>) -> rule that were changed to flip class.
                new_class (string)                            -> new class as predicted by the SigDirect surrogate.
        """

        data_cls = self._surrogate_model.predict(data_row.to_numpy().reshape((1, -1)).copy())[0]
        #print("IAIN DATA CLASS SURROGATE: ", data_cls)
        #aa = self._surrogate_model.get_contrast_sets(data_row.copy(), raw_rules=True, max_dev=0.05,
        #                                             new_class=wanted_class)
        #print("IAIN CONTRAST ", aa)
        # self._counterfactual.fit(self._surrogate_model.get_all_rules(raw_rules=True),
        #                          self._surrogate_model.get_ohe_simple())


        original_enc = self._surrogate_model._encode(data_row.to_numpy().reshape((1, -1)).copy())
        counter_predict, counter_rules = self._counterfactual.predict(original_enc.copy(), data_cls,
                                                                      new_class=wanted_class,
                                                                      n_counterfactuals=n_counterfactuals)
        #print("IAIN getting prediction")
        #print(counter_predict)
        counter_all_rules = []
        counter_all_values = []
        counter_all_predict = []
        # counter_predict = self._surrogate_model._decode([counter_predict])
        for i in range(n_counterfactuals):
            counter_p = counter_predict[i]
            counter_r = counter_rules[i]
            counter_v = self._surrogate_model._decode([counter_p.copy()])
            #print("IAIN COUNTER: ", i, counter_v)
            #print("IAIN ORIGINAL PREDICTION: ", self._surrogate_model.predict(data_row.to_numpy().reshape((1, -1)).copy())[0])
            #print("IAIN COUNTER PREDICTION: ", self._surrogate_model.predict(np.array(counter_v)))
            # TODO: I think that sigdirect needs to segment images before training

            for i in range(len(original_enc[0])):
                if original_enc[0][i] == counter_p[i] and counter_p[i] == 1:
                    temp = np.zeros(len(original_enc[0])).astype(int)
                    temp[i] = 1
                    position = self._surrogate_model._decode(temp.reshape((1, -1)))[0]
                    ind_use = np.where(np.array(position) != None)[0][0]
                    #print("IAIN REPLACEMENT: ")
                    #print(ind_use)
                    #print(counter_v[0][ind_use])
                    #print(data_row.to_numpy().reshape(1, -1)[0][ind_use])
                    #print(data_row.to_numpy().reshape(1, -1))
                    counter_v[0][ind_use] = data_row.to_numpy().reshape(1, -1)[0][ind_use]

            # fix nans
            #print("IAIN BEFORE FIX: ", counter_v)
            for i in range(len(counter_v[0])):
                if counter_v[0][i] is None or (not isinstance(counter_v[0][i], str) and math.isnan(counter_v[0][i])):
                    counter_v[0][i] = data_row.to_numpy().reshape(1, -1)[0][i]

            new_class = self._surrogate_model.predict(np.array(counter_v))[0]
            #print("IAIN NEW VALUE: ", counter_v)
            #print("IAIN NEW CLASS: ", new_class)
            counter_all_predict.append(new_class)
            counter_all_rules.append([(self._surrogate_model.raw_rule_translation(a[0], a[1]),
                                       self._surrogate_model.raw_rule_translation(b, wanted_class)) for a, b in counter_r])
            counter_all_values.append(counter_v)
        #for a, b in counter_rules:
        #    print("IAIN RULES: ", a, b)
        return counter_all_values, counter_all_rules, counter_all_predict

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score, weights=None, original_data=None):
        if weights is not None and weights in 'euclidean':
            if comparison_data is None:
                weights = euclidean_weights(self._perturber._encoder.transform(original_data),
                                            self._perturber._encoder.transform(self._perturbed_data))
            else:
                weights = euclidean_weights(self._perturber._encoder.transform(original_data),
                                            self._perturber._encoder.transform(comparison_data).to_numpy())
        elif weights is not None and weights in 'nearest-neighbors':
            if comparison_data is None:
                weights = nearest_neighbor_weights(self._perturber._encoder.transform(original_data),
                                            self._perturber._encoder.transform(self._perturbed_data))
            else:
                weights = nearest_neighbor_weights(self._perturber._encoder.transform(original_data),
                                            self._perturber._encoder.transform(comparison_data).to_numpy())
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        if (comparison_model is None) and (comparison_data is None):
            return comparison_method(self._blackbox_classification['perturbed'],
                                     self._surrogate_classification['perturbed'],
                                     sample_weight=weights)
        #elif (comparison_model is None) and (comparison_data is not None):
        #    return comparison_method(self._blackbox_classification['perturbed'],
        #                             self._surrogate_classification['perturbed'],
        #                             sample_weight=weights)

        elif (comparison_model is not None) and (comparison_data is None):
            wrapped_comparison = BlackBoxWrapper(comparison_model)
            if self._input_sets_class:
                wrapped_comparison.set_class(self._blackbox_classification['input'])
            return comparison_method(wrapped_comparison.predict(self._perturbed_data),
                                     self._surrogate_classification['perturbed'],
                                     sample_weight=weights)
        elif (comparison_model is not None) and (comparison_data is not None):
            wrapped_comparison = BlackBoxWrapper(comparison_model)
            if self._input_sets_class:
                wrapped_comparison.set_class(self._set_class)
            if self._verbose:
                print(self._verbose_header, " BBModel: ", wrapped_comparison.predict(comparison_data),
                      " BARBE: ", self._surrogate_model.predict(comparison_data.to_numpy()))
            return comparison_method(wrapped_comparison.predict(comparison_data),
                                     self._surrogate_model.predict(comparison_data.to_numpy()),
                                     sample_weight=weights)

    def get_rules(self, applicable=None):
        """
        Input: applicable (None or pandas Series) -> applicable row to find rules for.
                | Default: None
        Purpose: Get all the rules that apply to applicable or all the rules if none is given.
        Output: list<(rule_text, support, confidence)>, all rules with their support and confidence.
        """
        if applicable is not None:
            return self._surrogate_model.get_applicable_rules(applicable.to_numpy().reshape((1, -1)))
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
        elif feature in 'covariance':
            return self._perturber.get_cov()
        elif feature in 'means':
            return self._perturber.get_means()
        return None

    def get_next_best_class(self):
        # pass the second most abundant class in the perturbed data to the input
        current_label = self._blackbox_classification['input']
        occurences = Counter(self._blackbox_classification['perturbed'])
        most_abundant_label = None
        for clabel in occurences.keys():
            if (most_abundant_label is None and
                (clabel != current_label and
                 occurences[clabel] > occurences[most_abundant_label])):
                most_abundant_label = clabel
        return most_abundant_label

    def get_features(self, input_data, true_label):
        """
        Input: input_data (pandas DataFrame row)    -> Input to get feature usage from.
               true_label (string, int, or boolean) -> True label that would be assigned by the black box model. Must be
                |                                       set to a boolean if using input_sets_class = True.
        Purpose: Get the features and their cumulative support from the surrogate model, negative values indicate a
                  rule is present but not used when predicting the input data.
        Output: list<(feature, sum of support)>, all features used when predicting the input data and the total support.
        """
        return self._surrogate_model.get_features(input_data, true_label, self.get_next_best_class())

    def get_categories(self):
        """
        Input: None.
        Purpose: Get the features that are defined as categorical by the surrogate model.
        Output: dict<(feature column number, list<obj> -> len<=10)>
        """
        self._surrogate_model.get_categories()

    def predict(self, X):
        return self._surrogate_model.predict(X)

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
            while fit_tries < 10 and not fit_success:
                fit_tries += 1
                temp_data = input_data.copy()
                fit_success = self._fit_surrogate_model(temp_data, input_model)
            if not fit_success:
                if not ignore_errors:
                    #print(self._blackbox_classification['input'])
                    #print(self._surrogate_classification['input'])
                    #print(confusion_matrix(self._blackbox_classification['perturbed'],
                    #                       self._surrogate_classification['perturbed']))
                    raise AssertionError('BARBE ERROR: model did not successfully match input data in 5 tries.')
                else:
                    return None
            if self._verbose:
                print(self._verbose_header, 'number of tries:', fit_tries, fit_success)

        if self._verbose:
            print(self._verbose_header, 'fidelity:', self.get_surrogate_fidelity())
        # expecting a fit model explain the result for the new data, unless refit it assumes the prediction is correct
        # explanation_temp = self.get_features(input_data, self._blackbox_classification['input'][0])
        self._explanation = self.get_features(input_data,
                                              self._surrogate_model.predict(input_data.to_numpy().reshape(1, -1))[0])

        return self._explanation.copy()
