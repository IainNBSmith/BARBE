"""
Perturbs input data based on different modes.

This function should have the same functionality that the LimeWrapper in lime_interface.py uses.
"""

import numpy as np
from numpy.random import Generator, PCG64
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from barbe.discretizer import CategoricalEncoder
import warnings
from sklearn.utils import check_random_state


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
               input_bounds (list<(floats)>)    -> list of bounds in order (min, max), (None, max), (min, None), or None
                |                                   setting bounds for each variable in the final output. Will be
                |                                   checked for values that will easily be exceeded.
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
               bounding_method (string)         -> When generating perturbed samples how should a value outside of 
                |                                   the bounds be returned to the bounds given.
                |                                   'distribute-mean'   -> redistribute the values outside bounds directly
                |                                                           around the mean of the distributions.
                |                                   'distribute-sample' -> redistribute values outside the bounds around
                |                                                           the sample used to generate perturbations.
                |                                   'absolute-bounds'   -> create a value between the bounds by adding 
                |                                                           or subtracting the remainder when divided by
                |                                                           the size of the bounds divided by 2.
                |                                   'lazy-bounds'       -> overflow values directly to the min or max.
                | Default: 'distribute-sample' Options: {'distribute-mean', 'distribute-sample', 'absolute-bounds', 
                |                                      'lazy-bounds'}
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

    def __init__(self, training_data=None,
                 input_feature_names=None, input_scale=None, input_covariance=None, input_means=None,
                 input_categories=None,
                 input_bounds=None,
                 perturbation_type='uniform', bounding_method='distribute-mean', dev_scaling_factor=1,
                 uniform_training_range=False, uniform_scaled=True, df=None, covariance_mode='full',
                 use_mean_categorical_odds=True, standardized_categorical_variance=0.3334,  # new is 0<float<1 or None
                 find_non_categorical_bounds=False,
                 random_seed=None):
        #standardized_categorical_variance = None
        # standardized_categorical_variance = None
        #print("PERTURBER INIT: ", dev_scaling_factor)
        self._input_mode = ""  # either training or premade
        if input_scale is not None:
            input_scale = np.array(input_scale)
        #self._check_input(training_data, input_scale, input_categories)
        # check and modify training data
        #print("IAIN IN LOOP")

        self._encoder = CategoricalEncoder(category_threshold=10,
                                           find_non_categorical_bounds=find_non_categorical_bounds)
        training_data = self._encoder.fit_transform(training_data=training_data,
                                                    initial_key=input_categories,
                                                    data_features=input_feature_names)
        self._covariance_mode = covariance_mode
        self._bounding_method = bounding_method
        self._n_features = training_data.shape[1] \
            if input_scale is None else len(input_scale)
        # if given use degrees of freedom, if not then use the shape of the data, if no data set default
        self._df = training_data.shape[0] \
            if df is None and training_data is None else df  # over 200 is normal dist
        self._df = 20 if self._df is None else self._df

        self._means = self._calculate_means(training_data) \
            if input_means is None and training_data is not None else \
            ([0 for _ in range(len(input_scale))] if input_means is None else input_means)  # required to recenter input

        # original value is considered the mean if not declared
        self._original_value = self._means

        # do not reduce the deviation if input_scale is given
        self._standardized_categorical_variance = standardized_categorical_variance
        self._scale = self._calculate_scale(training_data) \
            if input_scale is None else input_scale

        #  whether uniform data should be generated based on the training data range
        self._uniform_training_range = uniform_training_range
        # whether uniform perturbation should scale to deviation in training data (independent of training range)
        self._uniform_scaled = uniform_scaled
        self._max, self._min = self._calculate_range(training_data,
                                                     input_shape=len(input_scale) if input_scale is not None else None)

        # find and adjust variance values
        self._covariance = self._calculate_covariance(training_data) \
            if input_covariance is None and input_scale is None else \
            (input_covariance if input_covariance is not None else np.diag(input_scale))
        # self._covariance = self._covariance  # / (dev_scaling_factor ** (1 / 2))
        self._fix_categorical_variance()
        self._scale = self._scale / dev_scaling_factor
        self._covariance = self._covariance / dev_scaling_factor
        #if input_covariance is None and input_scale is None:
        #    self._covariance[np.diag_indices_from(self._covariance)] = \
        #        self._covariance[np.diag_indices_from(self._covariance)] * dev_scaling_factor  # ** (1 / 2))

        self._distribution = perturbation_type
        self._bounds = self._check_bounds(input_bounds) \
            if not find_non_categorical_bounds else self._encoder.get_bounds()
        self._use_mean_categorical_odds = use_mean_categorical_odds

        self._random_state = check_random_state(random_seed)

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

    def _check_bounds(self, input_bounds, fix_scale=False, check_mean=None):
        # IAIN include checking if the given bound is declared for a categorical value
        error_header = "Barbe Perturber Error"
        if input_bounds is None:
            return input_bounds
        if check_mean is None:
            check_mean = self._means
        try:
            if len(check_mean[0]) > 1:
                check_mean = check_mean[0]
        except:
            check_mean = check_mean
        fixed_bounds = input_bounds.copy()
        # check that bounds are appropriate
        for i in range(len(input_bounds)):
            bound = input_bounds[i]
            center = check_mean[i]
            min_deviation, max_deviation = self._get_dev_from_mode(i)
            if bound is not None:
                min_b, max_b = bound
                min_check = (center - min_deviation) < min_b if min_b is not None else False
                max_check = (center + max_deviation) > max_b if max_b is not None else False
                #print(check_mean)
                #print(check_mean[i])
                #print(center, min_deviation, min_b, max_b)
                #print(min_check, max_check)
                #print(i)
                if min_check or max_check:
                    warning_text = "minimum bound " + str(min_b) if min_check else ""
                    if min_check and max_check:
                        warning_text += " and "
                    warning_text += "maximum bound " + str(max_b) if max_check else ""

                    warnings.warn(error_header + " provided deviation may cause perturbations to regularly exceed " +
                                  "the " + warning_text + " of feature " + str(i) + " "
                                  "to correct either reduce scale with dev_scaling or modify bounds.")
                    print(error_header + " provided deviation may cause perturbations to regularly exceed " +
                                  "the " + warning_text + " of feature " + str(i) + " "
                                  "to correct either reduce scale with dev_scaling or modify bounds.")
        return fixed_bounds

    def _fix_categorical_variance(self):
        # makes the categorical variables use standardized variance if given
        if self._standardized_categorical_variance is not None:
            categorical_indices = self._encoder.get_categorical_indices()
            for i in categorical_indices:
                old_covariance_val = self._covariance[i, i]
                if self._standardized_categorical_variance == "means":
                    self._covariance[i, i] = self._means[i]
                else:
                    self._covariance[i, i] = self._standardized_categorical_variance**2
                for j in range(self._covariance.shape[0]):
                    if i != j:
                        new_cov_ij = ((self._covariance[i, i] * np.sqrt(self._covariance[j, j])) *
                                      (self._covariance[i, j] / np.sqrt(old_covariance_val * self._covariance[j, j])))
                        self._covariance[i, j] = new_cov_ij
                        self._covariance[j, i] = new_cov_ij
                        #if np.isnan(self._covariance[i, j]):
                        #    print("OH NO!")
                        #    print(old_covariance_val, self._standardized_categorical_variance)
                if self._standardized_categorical_variance == "means":
                    self._scale[i] = self._means[i]
                else:
                    self._scale[i] = self._standardized_categorical_variance

        return None

    def _get_dev_from_mode(self, i):
        # get a deviation value based on the mode used to produce perturbations
        if self._distribution in 'uniform':
            if not self._uniform_scaled:
                return self._min[i], self._max[i]
            return self._min[i]*self._scale[i], self._max[i]*self._scale[i]
        elif self._distribution in 'normal':
            return self._scale[i]*2, self._scale[i]*2
        elif self._distribution in 'cauchy':
            return self._scale[i]*100, self._scale[i]*100
        elif self._distribution in 't-distribution':
            return self._scale[i] * 5, self._scale[i] * 5
        return None

    def _conversion_input(self, input_array):
        self._original_value = input_array
        enc_og = self._encoder.transform(input_array)
        # TODO: when using means the bias is very important see about good default values
        #  TODO: so far 1 does seem to work well but does not make sense in application
        # TODO: best run so far was using means and bias of 1 with a good split between classes and 83% nearest acc
        if self._use_mean_categorical_odds:
            new_input = self._encoder.rescale_categorical(enc_og, means=self._means, current_category_bias="avg_means")
            #new_input = self._encoder.rescale_categorical(enc_og, means=self._means, current_category_bias=0)
        #    bias_added = new_input.to_numpy() - self._means  # NEW Jan 22
        #    categorical_indices = self._encoder.get_categorical_indices()
        #    for i in categorical_indices:
        #        if new_input[i] > (1+(1e-8)):
        #            for j in range(self._covariance.shape[0]):
        #                if i != j:
        #                    sqrtj_var = np.sqrt(self._covariance[j, j]) * np.sqrt(self._covariance[i, i])
        #                    #if np.isnan(sqrtj_var):
        #                    #    print("OH NO2!")
        #                    #    print(self._covariance[j, j], self._covariance[i, i])
        #                    self._covariance[i, j] -= bias_added[i]*(sqrtj_var/self._adjusting_factor[i])
        #                    self._covariance[j, i] -= bias_added[i]*(sqrtj_var/self._adjusting_factor[i])
            return new_input

        else:
            return self._encoder.rescale_categorical(enc_og,
                                                     means=np.zeros(len(self._encoder._encoder_feature_values)),
                                                     current_category_bias=0)

    def _perturbed_discrete_conversion(self, perturbed_array):
        # IAIN we may want to make this into a one hot instead i.e. 1,2,3,4 = 00, 01, 10, 11
        #  while still taking the scale for individual values (what we do is generate perturbations as +/- and round to
        #  1 or 0) -> we also may consider doing a harder system i.e. 1,2,3,4 = 0001, 0010, 0100, 1000 we
        #  may consider this instead because then we have an equal likelyhood of going to any value while having a
        #  higher chance to stay at the original value. For this reason we will likely go the one hot way.
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
        # IAIN for change this should consider all values from the perturbed array that represent that feature
        #  nearest value becomes simpler as we only take the highest value from among the changed values and use it.
        categorical_features = self._encoder.get_categorical_features()
        categorical_key = self._encoder.get_encoder_key()
        for feature in categorical_features:
            perturbed_array[:, i] = nearest_values(perturbed_array[feature],
                                                   [item for key, item in categorical_key[i].items()])
            replacement_values = np.array([None for i in range(perturbed_array.shape[0])])
            for dvalue in self._categorical_key[i].keys():
                replacement_values[(perturbed_array[:, i] == self._categorical_key[i][str(dvalue)])] = dvalue
            #print("IAIN ", i, perturbed_array.shape, len(self._feature_original_types))
            #print(replacement_values)
            #print(self._feature_original_types)
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
        #print(training_array)
        #print(training_array.dtypes)
        return np.nanstd(training_array.to_numpy().astype(np.float64), axis=0)

    def _calculate_means(self, training_array):
        return np.nanmean(training_array, axis=0)

    def _rescale_data(self, unscaled_data, scaling_mean=None):
        if scaling_mean is None:
            scaling_mean = self._means
        #print((unscaled_data * self._scale.T).shape)
        scaled_data = np.array(unscaled_data * self._scale) + np.array(scaling_mean)
        return scaled_data

    def _bounding_distribute(self, values, min_b, max_b, over_value):
        while not np.all(values >= min_b) and not np.all(values <= max_b):
            if min_b is not None:
                values[values < min_b] = over_value + (
                        values[values < min_b] % (max_b - min_b)) / 2
            if max_b is not None:
                values[values > max_b] = over_value - (
                        values[values > max_b] % (max_b - min_b)) / 2
        return values

    def _bounding_lazy(self, values, min_b, max_b):
        if min_b is not None:
            values[values < min_b] = min_b
        if max_b is not None:
            values[values > max_b] = max_b
        return values

    def _bounding_absolute(self, values, min_b, max_b):
        if min_b is not None:
            values[values < min_b] = max_b - (values[values < min_b] % (max_b - min_b))/2
        if max_b is not None:
            values[values > max_b] = min_b + (values[values > max_b] % (max_b - min_b)) / 2
        return values

    def _bound_data(self, scaled_data):
        if scaled_data is None:
            return scaled_data
        if self._bounds is not None:
            # fix the data based on the bounds
            for i in range(len(self._bounds)):
                bound = self._bounds[i]
                mean = self._means[i]
                if bound is not None:
                    min_b, max_b = bound
                    # IAIN TODO: give an option to the user to use different bounding techniques
                    #  also include these in results when explaining things to Osmar
                    # other version that boosts the center instead
                    # IAIN TODO: worth implementing and testing both versions as part of the tests
                    #  try a few different things
                    # IAIN TODO: make the perturber / BARBE learn bounds when used with training data
                    # IAIN use self._bounding_method
                    if self._bounding_method == 'distribute-mean':
                        scaled_data[:, i] = self._bounding_distribute(scaled_data[:, i], min_b, max_b, mean)
                    elif self._bounding_method == 'distribute-sample':
                        scaled_data[:, i] = self._bounding_distribute(scaled_data[:, i], min_b, max_b,
                                                                      self._original_value)
                    elif self._bounding_method == 'absolute-bounds':
                        scaled_data[:, i] = self._bounding_absolute(scaled_data[:, i], min_b, max_b)
                    elif self._bounding_method == 'lazy-bounds':
                        scaled_data[:, i] = self._bounding_lazy(scaled_data[:, i], min_b, max_b)

        return scaled_data

    def _calculate_covariance(self, training_array):
        #print(training_array)
        #np.nanstd(training_array.to_numpy().astype(np.float64), axis=0)
        full_cov = np.cov(training_array.to_numpy().T.astype(np.float64))
        #print("IAIN USED FOR COV", training_array)
        if self._covariance_mode in 'full':
            return full_cov
        elif self._covariance_mode in 'diagonal':
            return np.diag(np.diag(full_cov))
        return None

    def _fetch_perturbation_rows(self, row_array, num_perturbations):
        return_data = None
        if self._distribution in 'uniform':
            # low high size
            if self._uniform_training_range:
                return_data = self._encoder.inverse_transform(self._random_state.uniform(self._min, self._max, size=(num_perturbations, self._n_features)))
                return self._bound_data(return_data)

            if not self._uniform_scaled:
                return_data = self._encoder.inverse_transform((self._random_state.uniform(self._min, self._max, size=(num_perturbations, self._n_features)) +
                        row_array))
                return self._bound_data(return_data)

            return_data = self._encoder.inverse_transform(self._rescale_data(self._random_state.uniform(self._min, self._max,
                                                                 size=(num_perturbations, self._n_features)),
                                      scaling_mean=row_array))
        elif self._distribution in 'normal':
            # location scale size
            #print(self._covariance)
            #print("ROW TO PERTURB: ", row_array.to_numpy().reshape(1, -1))
            #print("COV: \n", self._covariance)
            return_data = self._encoder.inverse_transform(self._random_state.multivariate_normal(row_array.to_numpy().reshape(1, -1).astype(np.float64)[0],
                                                                                                 self._covariance,
                                                                                                 size=num_perturbations)
                                                          )
        elif self._distribution in 'cauchy':
            # size (requires scaling)
            return_data = self._encoder.inverse_transform(self._rescale_data(self._random_state.standard_cauchy(size=(num_perturbations, self._n_features)),
                                      scaling_mean=row_array))
        elif self._distribution in 't-distribution':
            # df size
            return_data = self._encoder.inverse_transform(self._rescale_data(self._random_state.standard_t(self._df, size=(num_perturbations,
                                                                                    self._n_features)),
                                      scaling_mean=row_array))

        elif self._distribution in 'standard-normal':
            return_data = self._encoder.inverse_transform(
                self._rescale_data(self._random_state.normal(0, 1, size=(num_perturbations, self._n_features)),
                                   scaling_mean=row_array))

        return self._bound_data(return_data)

    def get_discrete_values(self):
        return self._encoder.get_encoder_key()

    def get_scale(self):
        return self._scale.copy()

    def get_means(self):
        return self._means.copy()

    def get_cov(self):
        return self._covariance.copy()

    def produce_perturbation(self, num_perturbations, data_row=None):
        if data_row is None:
            data_row = self._means
        else:
            data_row = self._conversion_input(data_row)
        self._check_bounds(self._bounds, check_mean=data_row)
        # returns perturbed data
        perturbed_data = self._fetch_perturbation_rows(data_row, num_perturbations)
        perturbed_data.iloc[0] = self._original_value  # make sure to include the row in training
        #perturbed_data = self._encoder.inverse_transform(perturbed_data)
        #print(perturbed_data.iloc[0:10])
        #print(self._means)
        #print(self._scale)
        #assert False
        return perturbed_data


class ClassBalancedPerturber(BarbePerturber):
    def __init__(self, balanced_threshold=0.01,
                 max_iterations=10,
                 balance_mode='curr-other',
                 **kwargs):
        BarbePerturber.__init__(self, **kwargs)
        self._iterations = 0
        self._max_iterations = max_iterations
        self._classes = None
        self._balance_tolerance = balanced_threshold
        self._num_perts = None
        self._balance_mode = balance_mode  # pure-balance, curr-other
        self._current_class = None
        self._final_balance = None

    def _get_class_counts(self, pert_classes):
        if self._classes is None:
            self._classes = np.unique(pert_classes)

        pert_classes = np.array(pert_classes)
        class_counts = {}
        for c in self._classes:
            class_counts[c] = (np.sum(pert_classes == c) / self._num_perts)
        return class_counts

    def _check_threshold_balance(self, class_counts):
        # check if the value is within tolerance of the even threshold for the smallest amount of data
        print("Checking: ", class_counts)
        if self._balance_mode != 'curr-other':
            n_total_classes = len(self._classes)
            for c, proportion in class_counts.items():
                proportion_comparison = ((1 / n_total_classes) - self._balance_tolerance) \
                    if ((1 / n_total_classes) > self._balance_tolerance) \
                    else (1 / n_total_classes)
                print("My check: ", proportion_comparison, proportion)
                if proportion < proportion_comparison:
                    return False
            return True
        else:
            total_other = 0
            total_current = 0
            for c, proportion in class_counts.items():
                if c == self._current_class:
                    total_current = proportion
                else:
                    total_other += proportion
            balance_check = lambda x: x >= 0.5 - self._balance_tolerance
            #print("My check: ", total_current, total_other)
            #print("My check: ", balance_check(total_current), balance_check(total_other))
            return balance_check(total_current) and balance_check(total_other)

    def _undersample_classes(self, pert_data, pert_classes):
        selected_rows = np.array([])
        balance_dict = {}
        if self._balance_mode != 'curr-other':
            for c in self._classes:
                satisfied_pos = np.argwhere(pert_classes == c).reshape((1, -1))[0]
                choice_size = int(self._num_perts//len(self._classes))
                # only select as much data as there is in the dataset
                choice_size = choice_size if len(satisfied_pos) >= choice_size else len(satisfied_pos)
                selected_rows = np.append(selected_rows, np.random.choice(satisfied_pos, size=choice_size, replace=False))
                balance_dict[c] = choice_size
        else:
            total_other_classes = np.sum(pert_classes != self._current_class)
            for c in self._classes:
                satisfied_pos = np.argwhere(pert_classes == c).reshape((1, -1))[0]
                if c == self._current_class:
                    choice_size = int(self._num_perts//2)
                else:
                    # keep the proportions within the others the same, just get them to the right amount
                    choice_size = int(((self._num_perts/2) * (len(satisfied_pos)/total_other_classes)) // 1)
                # only select as much data as there is in the dataset
                choice_size = choice_size if len(satisfied_pos) >= choice_size else len(satisfied_pos)
                selected_rows = np.append(selected_rows,
                                          np.random.choice(satisfied_pos, size=choice_size, replace=False))
                balance_dict[c] = choice_size

        self._final_balance = balance_dict
        return pert_data.iloc[selected_rows]

    def get_number_iterations(self):
        return self._iterations

    def get_balance(self):
        return self._final_balance.copy()

    def produce_balanced_perturbation(self, num_perturbations, bbmodel, data_row=None):
        perturbed_data = self.produce_perturbation(num_perturbations, data_row=data_row)
        pert_classes = bbmodel.predict(perturbed_data)
        if self._balance_mode == 'curr-other':
            self._current_class = pert_classes[0]  # first will be input from produce_perturbation
        self._classes = np.unique(pert_classes)
        self._num_perts = num_perturbations
        class_counts = self._get_class_counts(pert_classes)
        self._iterations = 0
        while self._iterations < self._max_iterations and not self._check_threshold_balance(class_counts):
            new_perturbed = self.produce_perturbation(num_perturbations, data_row=data_row)
            perturbed_data = perturbed_data.append(new_perturbed, ignore_index=True)
            new_classes = bbmodel.predict(new_perturbed)
            pert_classes = np.append(pert_classes, new_classes)
            class_counts = self._get_class_counts(pert_classes)
            self._iterations += 1
        # error message if max iterations hit?
        perturbed_data = self._undersample_classes(perturbed_data, pert_classes)
        perturbed_data.iloc[0] = self._original_value
        return perturbed_data


class ClusteredPerturber(BarbePerturber):
    def __init__(self, training_data, clustering_algorithm='kmeans', n_clusters=4, assigned_clusters=None,
                 input_bounds=None, perturbation_type='uniform', covariance_mode='full', uniform_training_range=False,
                 uniform_scaled=True, dev_scaling_factor=1, df=None, random_seed=None):
        BarbePerturber.__init__(self, training_data=training_data, input_bounds=input_bounds,
                                perturbation_type=perturbation_type, covariance_mode=covariance_mode,
                                uniform_training_range=uniform_training_range, uniform_scaled=uniform_scaled,
                                dev_scaling_factor=dev_scaling_factor, df=df, random_seed=random_seed)
        training_data = training_data.to_numpy()
        self._clustering_algorithm = clustering_algorithm
        self._n_clusters = n_clusters
        clusters = self._cluster(training_data) if assigned_clusters is None else assigned_clusters
        self._req_in_cluster = False if assigned_clusters is None else True
        self._cluster_base = np.min(clusters)
        self._generate_cluster_values(training_data, clusters)

        # calculate means for each cluster

    def _cluster(self, training_data):
        if self._clustering_algorithm in 'kmeans':
            self._clusterer = KMeans(n_clusters=self._n_clusters).fit(training_data)

        elif self._clustering_algorithm in 'mixtures':
            self._clusterer = GaussianMixture(n_components=self._n_clusters, covariance_type=self._covariance_mode).fit(training_data)

        return self._clusterer.predict(training_data)

    def _generate_cluster_values(self, training_data, clusters):
        # IAIN sets some general values to each cluster
        self._all_means = []
        self._all_scales = []
        self._all_cov = []
        self._all_ranges = []
        for i in range(self._n_clusters):
            relevant_data = training_data[np.where(clusters == (i-self._cluster_base))[0], :]
            #print(np.where(clusters == (i-self._cluster_base)))
            #print(relevant_data.shape)
            #print(training_data.shape)
            self._all_means.append(self._calculate_means(relevant_data))
            self._all_scales.append(self._calculate_scale(relevant_data))
            self._all_cov.append(self._calculate_covariance(relevant_data))
            self._all_ranges.append(self._calculate_range(relevant_data))

    def _set_cluster_values(self, data_row=None, data_cluster=None):
        if data_row is not None and data_cluster is None:
            if self._req_in_cluster and data_cluster is None:
                raise ValueError("Error: perturber requires a cluster and none were provided.")
            data_cluster = self._clusterer.predict(data_row.reshape(1, -1))[0]
        # set the values used in generating perturbations to that of the given row
        self._means = self._all_means[data_cluster-1]
        self._scale = self._all_scales[data_cluster-1]
        self._covariance = self._all_cov[data_cluster-1]
        self._max, self._min = self._all_ranges[data_cluster-1]

    def produce_perturbation(self, num_perturbations, data_row=None, data_cluster=None):
        if data_row is None and data_cluster is None:
            data_row = self._all_means[0]
        else:
            data_row = self._conversion_input(data_row)

        self._set_cluster_values(data_row=data_row)
        if data_row is None and data_cluster is not None:
            data_row = self._means

        self._check_bounds(self._bounds, check_mean=data_row)
        # returns perturbed data
        perturbed_data = self._fetch_perturbation_rows(data_row, num_perturbations)
        perturbed_data[0, :] = data_row  # make sure to include the row in training
        perturbed_data = self._perturbed_discrete_conversion(perturbed_data)
        return perturbed_data