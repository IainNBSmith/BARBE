# DONE/TODO: separate BARBE code from LIME (Jun 7)
# DONE/TODO: make the code called in experiments_barbe.py call to barbe.py (Jun 11)
# DONE/TODO: find locations where lime is used (Jun 11)
# TODO: create relevant comments indicating execution of BARBE code
# TODO: modify header comments to my preferred style
# TODO: ensure that all the options that are possible function correctly (save this to a test file)
'''
 Done/TODO: make the code actually use and run sigdirect (it was calling base lime from what I could tell) (Jun 11)
    - Manually changed settings to set barbe_mode and model_regressor to 'BARBE' and SigDirect()
      in future these should be obvious defaults or removed entirely to avoid confusion since
      the other modes are directly lime code.
    - Add a TODO: remove lime code that will never run with these settings (avoid issues with having lime code)
        - This has a ways to go as much code is used in options that may never be applied, need to check this.
        - Add a TODO: is the explain_instance function ever used or is it only explain_instance_with_data? (remove?)
        - Add a TODO: make the code only call functions from within the object if possible (see get_all_rules)

 TODO: set up the skeleton (in another file) of my task, smallest feature change collection
'''

from __future__ import print_function

"""
Contains abstract functionality for learning locally linear sparse model.
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state

# eventually these must be modified into our own versions of the code
#  that goal is for the end of summer
# IAIN I can write this better (all lime Discretizers)
from lime.discretize import QuartileDiscretizer
from lime.discretize import DecileDiscretizer
from lime.discretize import EntropyDiscretizer
from lime.discretize import BaseDiscretizer
from lime.discretize import StatsDiscretizer
# IAIN do not know if this is lime or not
from lime import explanation
from . import lime_base

import os
import pickle
import itertools
from collections import Counter, defaultdict

import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

from sigdirect import SigDirect
# IAIN need to check what this is and who wrote it
# IAIN used for discretizing??? if so this should definitely be rewritten
from lime_tabular import TableDomainMapper

# the next two functions seem like they would be better implemented as part of the explainer
def get_all_rules(neighborhood_data, labels_column, clf):
    print('CALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
    clf.fit(neighborhood_data, labels_column)
    local_pred = clf.predict(neighborhood_data[0].reshape((1, -1)), 2).astype(int)[0]

    all_rules = clf.get_all_rules()
    return all_rules, local_pred


def get_features_sigdirect(all_rules, true_label):
    """
    Input: all_rules ()  ->
           true_label () ->
    Purpose: use applied rules first, and then the rest of the applicable rules, and then all rules (other labels,
     rest of them match)
    Output: feature_value_pairs () ->
    """

    # applied rules,
    applied_sorted_rules = sorted(all_rules[true_label],
                                  key=lambda x: (
                                      len(x[0].get_items()),
                                      - x[0].get_confidence() * x[0].get_support(),
                                      x[0].get_log_p(),
                                      - x[0].get_support(),
                                      -x[0].get_confidence(),
                                  ),
                                  reverse=False)

    # applicable rules, except the ones in applied rules.
    applicable_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x != true_label]),
                                     key=lambda x: (
                                         len(x[0].get_items()),
                                         - x[0].get_confidence() * x[0].get_support(),
                                         x[0].get_log_p(),
                                         - x[0].get_support(),
                                         -x[0].get_confidence(),
                                     ),
                                     reverse=False)

    # all rules, except the ones in applied rules.
    other_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x != true_label]),
                                key=lambda x: (
                                    len(x[0].get_items()),
                                    - x[0].get_confidence() * x[0].get_support(),
                                    x[0].get_log_p(),
                                    - x[0].get_support(),
                                    -x[0].get_confidence(),
                                ),
                                reverse=False)

    counter = len(all_rules)
    bb_features = defaultdict(int)

    # First add applied rules
    applied_rules = []
    for rule, ohe, original_point_sd in applied_sorted_rules:
        temp = np.zeros(original_point_sd.shape[0]).astype(int)
        temp[rule.get_items()] = 1
        if np.sum(temp & original_point_sd.astype(int)) != temp.sum():
            continue
        else:
            applied_rules.append(rule)
        rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
        #         rule_items = temp ## TEXT (uncomment for TEXT)
        for item, val in enumerate(rule_items):
            if val is None:
                continue
            #                 if val==0: ## TEXT (uncomment for TEXT)
            #                     continue ## TEXT (uncomment for TEXT)
            #                 if item not in bb_features:
            bb_features[item] += rule.get_support()
        #                     bb_features[item] += counter
        #                 bb_features[item] = max(bb_features[item],  rule.get_confidence()/len(rule.get_items()))
        counter -= 1
    set_size_1 = len(bb_features)

    # Second, add applicable rules
    applicable_rules = []
    for rule, ohe, original_point_sd in applicable_sorted_rules:
        temp = np.zeros(original_point_sd.shape[0]).astype(int)
        temp[rule.get_items()] = 1
        if np.sum(temp & original_point_sd.astype(int)) != temp.sum():
            continue
        else:
            applicable_rules.append(rule)
        rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
        #         rule_items = temp ## TEXT (uncomment for TEXT)
        for item, val in enumerate(rule_items):
            if val is None:
                continue
            if item not in bb_features:
                #                 bb_features[item] += rule.get_support()
                bb_features[item] += counter
        counter -= 1

    # Third, add other rules.
    other_rules = []
    for rule, ohe, original_point_sd in other_sorted_rules:
        temp = np.zeros(original_point_sd.shape[0]).astype(int)
        temp[rule.get_items()] = 1
        # avoid applicable rules
        if np.array_equal(temp, temp & original_point_sd.astype(int)):  # error??? it was orig...[0].astype
            continue
        #             elif temp.sum()==1:
        #                 continue
        elif temp.sum() - np.sum(temp & original_point_sd.astype(int)) > 1:  # error???
            continue
        #             else:
        rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
        #         rule_items = temp ## TEXT (uncomment for TEXT)
        seen_set = 0
        for item, val in enumerate(rule_items):
            if val is None:
                continue
            if item not in bb_features:
                #                 bb_features[item] += rule.get_support()
                #                     bb_features[item] += counter
                candid_feature = item
                pass
            else:
                seen_set += 1
        if seen_set == temp.sum() - 1:  # and (item not in bb_features):
            bb_features[candid_feature] += counter
            other_rules.append(rule)
        counter -= 1

    feature_value_pairs = sorted(bb_features.items(), key=lambda x: x[1], reverse=True)

    return feature_value_pairs, None


class BarbeBase(object):
    """
    Purpose: Class for learning association rules indicating important features via SigDirect. Model learns local
     feature importance through perturbing the input sample.
    """

    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Input:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from SigDirect model. TODO: verbose
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        # see what this function does
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """
        Input: weighted_data ()  -> data that has been weighted by kernel
               weighted_label () -> labels, weighted by kernel
        Purpose: Generates the lars path for weighted data.
        Output: alphas () ->
                coefs ()  -> both are arrays corresponding to the regularization parameter and coefficients,
                             respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    # IAIN check if these are actually used as the Ridge regressor should not be used right now.
    def forward_selection(self, data, labels, weights, num_features):
        '''
        Input: data ()         ->
               labels ()       ->
               weights ()      ->
               num_features () ->
        Purpose: Iteratively adds features to the model
        Output: used_features (numpy Array) ->
        '''
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        '''
        Input: data ()         ->
               labels ()       ->
               weights ()      ->
               num_features () ->
               method ()       ->
        Purpose: Selects features for the model. see explain_instance_with_data to understand the parameters.
        Output: (numpy Array) ->
        '''
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            # IAIN this is why we do the testing, to see what we can actually use...
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   # IAIN manually set this to always use SigDirect
                                   model_regressor=SigDirect(),
                                   neighborhood_data_sd=None,
                                   ohe=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        # IAIN we may remove feature_selection and associated code as this is not used...
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        # IAIN removed if and made it into just setting model_regressor here
        # IAIN TODO: should make this have a notable verbose mode
        model_regressor = SigDirect()
        all_rules = defaultdict(list)
        true_label = neighborhood_labels[0].argmax()
        labels_column = np.argmax(neighborhood_labels, axis=1)
        # IAIN model regressor here is sigdirect (within BarbeBase so it should have its own model saved in it)
        all_raw_rules, predicted_label = get_all_rules(neighborhood_data_sd, labels_column, model_regressor)

        # convert raw rules to rules (one-hot-decoding them)
        if predicted_label == true_label:
            for x, y in all_raw_rules.items():
                all_rules[x] = [(t, ohe, neighborhood_data_sd[0]) for t in y]
        else:
            predicted_label = -1  # to show we couldn't predict it correctly

        feature_value_pairs, prediction_score = get_features_sigdirect(all_rules, true_label)
        return (0, feature_value_pairs, prediction_score, predicted_label)


# replace instances of lime tabular explainer with BarbeExplainer
class BarbeExplainer(object):
    '''
    Purpose: Explains predictions on tabular data using Sigdirect.

    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained.
    '''

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        print('************** REGRET *********************')
        print('training_data Length = ', len(training_data))
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats

        # Check and raise proper error in stats are supplied in non-descritized path
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        print('************** REGRET ********************* feature_names', self.feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                # IAIN lime code
                discretizer = StatsDiscretizer(training_data, self.categorical_features,
                                               self.feature_names, labels=training_labels,
                                               data_stats=self.training_data_stats)

            if discretizer == 'quartile':
                # IAIN lime code
                self.discretizer = QuartileDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels)
            elif discretizer == 'decile':
                # IAIN lime code
                self.discretizer = DecileDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels)
                print('Hi There: ', self.discretizer)
            elif discretizer == 'entropy':
                # IAIN lime code
                self.discretizer = EntropyDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))
            print('self.categorical_features = ', self.categorical_features)

            # Get the discretized_training_data when the stats are not provided
            if (self.training_data_stats is None):
                # print('training_data = ', training_data)
                discretized_training_data = self.discretizer.discretize(training_data)
                # print('discretized_training_data = ', discretized_training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        # IAIN again worth checking what is used...
        self.feature_selection = feature_selection
        print('self.feature_selection = ', self.feature_selection)
        self.base = BarbeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names
        print('self.class_names = ', self.class_names)

        # Though set has no role to play if training data stats are provided
        self.scaler = None
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:  ## This code block runs
                    column = discretized_training_data[:, feature]
                else:
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

        print('after = ', training_data)

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):
        """
        Input: training_data_stats () ->
        Purpose: Method to validate the structure of training data stats
        Output: None, error on failed validation.
        """
        stat_keys = list(training_data_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise Exception("Missing keys in training_data_stats. Details:" % (missing_keys))

    # IAIN does seem to be used by the experiment_barbe file
    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         # IAIN never not use sigdirect
                         model_regressor=SigDirect(),
                         # IAIN the default is never reset so the old never called BARBE
                         barbe_mode='BARBE'):
        print('explain_instance', data_row,
              predict_fn,
              labels,
              top_labels,
              num_features,
              num_samples,
              distance_metric,
              model_regressor,
              barbe_mode)
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):  # code does not go here
            # Preventative code: if sparse, convert to csr format if not in csr format already
            print('okkkkkkkkkkkkkkk')
            data_row = data_row.tocsr()

        # IAIN define the sd_data that will be used later
        data, inverse, sd_data, ohe = self.__data_inverse_barbe(data_row, num_samples, predict_fn, barbe_mode)

        if sp.sparse.issparse(data):  # code does not go here
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        if barbe_mode == "BARBE":
            yss = predict_fn(inverse)
        elif barbe_mode == "TEXT":
            yss = predict_fn(sd_data)  ## TEXT
        else:
            # this should never happen
            raise ValueError("no barbe mode selected")

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("BARBE does not currently support "
                                          "classifier models without probability "
                                          "scores.")
            elif len(yss.shape) == 2:  # code comes here
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                print('self.class_names', self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    print('Arrived ???')
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        print('feature_names', feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:  # code comes here
            print('I guess')
            values = self.convert_and_round(data_row)
            feature_indexes = None

        print(values, feature_indexes)

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features
        print('categorical_features', categorical_features)

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        # IAIN lime code that we seem not to change
        print("IAIN using TableDomainMapper")  # it does use this so we must see
        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        # IAIN lime?
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            # IAIN what is yss?
            print("IAIN called from with data")
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
                # IAIN sd_data is the modified data
                neighborhood_data_sd=sd_data,
                ohe=ohe)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    # IAIN removed original __data_inverse (did not seem to be used)
    def __data_inverse_barbe(self,
                             data_row,
                             num_samples,
                             predict_fn,
                             barbe_mode=None):
        # IAIN this may indicate some things to remove
        '''
        Input: data_row (numpy Array (,n))           -> 1d numpy Array corresponding to a row.
               num_samples (int [1, inf))            -> size of the neighborhood to learn Sigdirect model.
               predict_fn ()                         ->
               barbe_mode (string {'BARBE', 'TEXT'}) -> whether to run numerical barbe or text barbe on data.
        Purpose: Generates a neighborhood around a prediction for BARBE.
        Output: data ( (num_samples,K))    -> data with encoded categorical features (0, 1) for match and non-match to
            `                                 row. First row is original instance.
                inverse ( (num_samples,K)) -> same as data but categorical features are not binary.
                sd_values ()               -> data used by Sigdirect.
                ohe ()                     -> A sklearn.preprocessing.OneHotEncoder object fit on the train data
        '''
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]
            data = self.random_state.normal(
                0, 1, num_samples * num_cols).reshape(
                num_samples, num_cols)
            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()

        if barbe_mode == 'BARBE':
            inverse = np.zeros((int(num_samples), first_row.shape[0]))
            for column in categorical_features:
                values = self.feature_values[column]
                proximities = np.abs(np.subtract(first_row[column], values))
                freqs = sp.special.softmax(- 0.5 * proximities)

                inverse_column = self.random_state.choice(values, size=inverse.shape[0],
                                                          replace=True, p=freqs)
                inverse[:, column] = inverse_column
            unique_count = np.unique(inverse, axis=0).shape[0]

            sd_values = np.array(inverse).astype(int)
            sd_values[0] = first_row
            if self.discretizer is not None:
                inverse[1:] = self.discretizer.undiscretize(inverse[1:])
            inverse[0] = data_row
            ohe = sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')
            # IAIN this is where the modification of data occurs
            sd_values = np.asarray(ohe.fit_transform(sd_values).todense()).astype(int)
            return data, inverse, sd_values, ohe
        elif barbe_mode == 'TEXT':
            # IAIN for later this may be what I change to import other stuff for BARBIE
            sd_values = np.zeros((int(num_samples), first_row.shape[0]))
            for idx in np.nonzero(first_row)[0]:
                t = np.random.choice((0, 1), size=num_samples)
                sd_values[:, idx] = t
            sd_values[0] = first_row
            return data, data, sd_values, None

        raise NameError("Wrong value for barbe_mode:", barbe_mode)
