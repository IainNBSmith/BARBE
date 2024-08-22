"""
Perturbs input data based on different modes.

This function should have the same functionality that the LimeWrapper in lime_interface.py uses.
"""

import numpy as np
from numpy.random import Generator, PCG64
import warnings


class BarbePerturber:
    __doc__ = '''
        Purpose: Perturbs input data or a given scale from BARBE into multiple samples.

        Input: training_data (pandas DataFrame) -> training data to find scales for making perturbations.
                | Default: None
               input_scale (list<float>)        -> scales of expected data if not given as training.
                | Default: None
               input_categories (dict<list>) or -> indicator for which values are categorical and the possible values.
                |                (dict<dict>)       or direct assignment of values to labels.
                | Default: None
               perturbation_type (string)       -> The type of distribution to use when generating perturbed data.
                |                                   'uniform' -> uniform distribution over a range (-2, 2) is equally as
                |                                                 likely to generate 0 as 0.1 as 1.2
                |                                   'normal' -> normal distribution, data will be more similar to the
                |                                                true distribution of the data along with all
                |                                                interactions between features.
                |                                   'cauchy' -> cauchy distribution, long-tailed distribution that
                |                                                captures more radical differences in feature values.
                |                                                Useful when edge cases are a concern.
                |                                   't-distribution' -> t distribution, useful when less training data
                |                                                        is available (for example if privacy is a 
                |                                                        concern). Has wider tails and more flexibility
                |                                                        than the normal distribution.
                | Default: 'uniform' Options: {'uniform', 'normal', 'cauchy', 't-distribution'}
               covariance_mode (string)         -> Used when perturbation_type = 'normal'. Covariance can either take
                |                                   all interactions into account or only deviation within a feature.
                | Default: 'full' Options: {'full', 'diagonal'}
               uniform_training_range (boolean) -> Used when peturbation_type = 'uniform'. Whether the uniform data is
                |                                   used to generate data somewhere in the range of the original input.
               uniform_scaled (boolean)         -> Used when peturbation_type = 'uniform'. Independent from
                |                                   uniform_training_range. Whether to scale uniform data so the
                |                                   deviations appear similar to the scale of the original data.
                |                                   Note: sometimes perturbation by itself is useful when model
                |                                   performance on edge cases is a concern.
               dev_scaling_factor (int>0)       -> Amount to scale the deviations (standard deviation and convariance).
                |                                   Makes the model tighter so new data is more similar to input.
               df (None or int>2)               -> Used when perturbation_type = 't-distribution'. Degrees of freedom 
                |                                   used when generating a t-distribution, will be set to the amount of
                |                                   training data if df = None. Note: When df > 100 t-distributions are
                |                                   similar to a normal distribution. Default is 20 if not given and the\
                |                                   input_scale is used.
               random_seed (int)                -> Random seed to use when generating perturbations. If None use a
                |                                   pseudo random state.
                | Default: None
        '''

    def __init__(self, training_data=None, input_scale=None, input_categories=None, perturbation_type='uniform',
                 covariance_mode='full', uniform_training_range=False, uniform_scaled=True, dev_scaling_factor=1,
                 df=None, random_seed=None):
        self._input_mode = ""  # either training or premade
        if input_scale is not None:
            input_scale = np.array(input_scale)
        self._check_input(training_data, input_scale, input_categories)
        # check and modify training data
        print("IAIN IN LOOP")
        if input_categories is None or (not input_categories and training_data is not None):
            self._categorical_features = []  # indicators of which columns are categorical
            self._feature_original_types = {}  # indicates the original type of all categorical values (supplied if given input_categories)
            self._categorical_key = dict()
            print("IAIN discrete conversion")
            training_data = self._training_discrete_conversion(training_data.to_numpy())
        else:
            print("IAIN NOT RIGHT", input_categories)
            self._feature_original_types = {}
            self._categorical_key = input_categories
            self._categorical_features = list(input_categories.keys())
            if any([isinstance(self._categorical_key[key], list) for key in self._categorical_key.keys()]):
                for key in self._categorical_key.keys():
                    if isinstance(self._categorical_key[key], list):
                        temp_replacement = dict()
                        count = 0
                        for item in self._categorical_key[key]:
                            temp_replacement[str(item)] = count
                            count += 1
                        self._categorical_key[key] = temp_replacement
            #self._feature_original_types = {}
            for key in input_categories.keys():
                print("IAIN POTENTIAL ISSUE: ", input_categories)
                self._feature_original_types[key] = type(list(input_categories[key].keys())[0])
            #self._feature_original_types = [type(input_categories[key][list(input_categories[key].keys())[0]])
            #                                for key in input_categories.keys()]

        print("IAIN I CAN PRINT")
        self._covariance_mode = covariance_mode
        self._n_features = training_data.shape[1] \
            if input_scale is None else len(input_scale)
        # if given use degrees of freedom, if not then use the shape of the data, if no data set default
        self._df = training_data.shape[0] \
            if df is None and training_data is None else df  # over 100 is normal dist
        self._df = 20 if self._df is None else self._df

        self._means = self._calculate_means(training_data) \
            if input_scale is None else [0 for _ in range(len(input_scale))]  # required to recenter input

        # do not reduce the deviation if input_scale is given
        self._scale = self._calculate_scale(training_data) / dev_scaling_factor \
            if input_scale is None else input_scale
        #  whether uniform data should be generated based on the training data range
        self._uniform_training_range = uniform_training_range
        # whether uniform perturbation should scale to deviation in training data (independent of training range)
        self._uniform_scaled = uniform_scaled
        self._max, self._min = self._calculate_range(training_data,
                                                     input_shape=len(input_scale) if input_scale is not None else None)
        # IAIN potentially add non-diagonal input scale values in future
        self._covariance = self._calculate_covariance(training_data) / dev_scaling_factor \
            if input_scale is None else np.diag(input_scale)

        self._distribution = perturbation_type
        self._random_state = Generator(PCG64()) \
            if random_seed is None else np.random.default_rng(seed=random_seed)

    def _check_input(self, input_data, input_scale, input_categories):
        # IAIN check that at either input data is not none or scale and categories are both not none
        # IAIN give error message or warning in some cases to note that category scales should be 1-2 depending on the number
        error_header = "BARBE Perturber Error"
        if input_data is None:
            # check input scale and input_categories
            # check that all the keys from input_categories are valid indices of input_scale
            #  if not then pass the error message and recommend scale depending on the # of uniques
            #  throw a warning if the scale is very small on a particular category
            # check that categories have relatively large scales
            for key in input_categories:
                temp_key = int(key)
                n_values = len(input_categories[key]) \
                    if isinstance(input_categories[key], list) else len(list(input_categories[key].keys()))
                if input_scale[temp_key] < n_values / 4:
                    warnings.warn(error_header + " scale may be too LOW to encounter variety of values in column:" +
                                  str(temp_key) + "\nsuggested to use a value near " + str(int(n_values / 4)) +
                                  "\nto avoid perturbations limited to the given value.")
                if input_scale[temp_key] > n_values * 4:
                    warnings.warn(error_header + " scale may be too HIGH to encounter variety of values in column: " +
                                  str(temp_key) + "\nsuggested to use a value near " + str(int(n_values * 4)) +
                                  "\nto avoid perturbation limited to the edge values.")
        else:
            # IAIN check the input data that it is a numpy or can be converted
            pass

    def _training_discrete_conversion(self, training_array, category_threshold=10):
        # conversion from discrete values -> numeric values
        for i in range(training_array.shape[1]):
            unique_values = list(np.unique(training_array[:, i].astype(str)))
            try:
                unique_values.remove('nan')
            except ValueError:
                pass
            print("IAIN UNIQUES ", unique_values)
            if (len(unique_values) <= category_threshold and
                    not np.all(np.isreal(list(training_array[:, i])))):
                self._categorical_features.append(i)
                print("UNIQUES IAIN: ", type(unique_values[0]))
                self._feature_original_types[i] = type(unique_values[0])
                # self._feature_original_types.append(type(unique_values[0]))
                self._categorical_key[i] = dict()
                for j in range(len(unique_values)):
                    value = str(unique_values[j])
                    self._categorical_key[i][value] = j
                    training_array[((training_array[:, i]).astype(str) == value), i] = j

        return training_array.astype(float)

    def _conversion_input(self, input_array):
        for i in self._categorical_features:
            # IAIN ERROR ORIGINATES HERE
            try:
                input_array[i] = self._categorical_key[i][str(input_array[i])]
            except Exception as e:
                raise ValueError(str(self._categorical_key) + " " +
                                 str(i) + " " + str(input_array) + " " + str(self._categorical_features) + " " + str(e))
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
            print("IAIN ", i, perturbed_array.shape, len(self._feature_original_types))
            print(replacement_values)
            print(self._feature_original_types)
            perturbed_array[:, i] = replacement_values.astype(self._feature_original_types[i])
        return perturbed_array

    def _calculate_range(self, training_array, input_shape=None):
        input_shape = training_array.shape[1] if input_shape is None else input_shape

        if self._uniform_training_range:
            return np.max(training_array, axis=0), np.min(training_array, axis=0)
        # default is essentially anywhere within two standard deviations if considering normal scale
        return (np.array([2 for _ in range(input_shape)]),
                np.array([-2 for _ in range(input_shape)]))

    def _calculate_scale(self, training_array):
        return np.nanstd(training_array, axis=0)

    def _calculate_means(self, training_array):
        return np.nanmean(training_array, axis=0)

    def _rescale_data(self, unscaled_data, scaling_mean=None):
        if scaling_mean is None:
            scaling_mean = self._means
        return (unscaled_data * list(self._scale)) + scaling_mean

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

            return self._rescale_data(self._random_state.uniform(self._min, self._max,
                                                                 size=(num_perturbations, self._n_features)),
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
            return self._rescale_data(self._random_state.standard_t(self._df, size=(num_perturbations,
                                                                                    self._n_features)),
                                      scaling_mean=row_array)
        return None

    def get_discrete_values(self):
        return self._categorical_key.copy()

    def get_scale(self):
        return self._scale

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
