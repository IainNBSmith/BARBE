import numpy as np
import pandas as pd


def check_numeric(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class CategoricalEncoder:
    __doc__ = '''
    Purpose: Encode data into numeric values that other systems can handle either given training data or pre-existing 
     information. Flexibly encode some values into one-hot vectors (categories with equal likelihood) or into ordinals
     (if the approach is deemed necessary). Handle sufficiently finite numeric values (based on user input) as numeric
     values that round to the nearest value to avoid mishandling numerics.

    Input: category_threshold (int>0) -> specifies the number of values that a numerical feature needs to have in order
           |                              to not be considered sufficiently finite or discrete and be treated as such.
           | Default: 10
    
    '''
    # ******** GLOBAL SETTINGS FOR CLASS ******** #
    # preferred setting are False, check if True is preferable in some scenarios
    _IGNORE_NUMERIC = False  # treat finite numeric values as any other categorical value
    _ORDINAL_ENCODING = False  # encode categorical values as numbers e.g. semi-urban=0, rural=1, urban=2

    DEFAULT_CAT_BIAS = {'current': lambda x: x/1.5,
                        'other': lambda x: 1/x}

    def __init__(self, category_threshold=10,
                 ordinal_encoding=_ORDINAL_ENCODING,
                 find_non_categorical_bounds=False):
        self._encoder_key = None
        self._original_feature_order = None  # how to return transformed data
        self._encoder_feature_values = None
        self._finite_numeric_features = None
        self._data_means = None  # used if setting unique starting values for categorical values
        self._current_category_bias = None
        self._category_threshold = category_threshold
        self._non_category_bounds = None

        self._find_non_category_bounds = find_non_categorical_bounds
        self._ordinal = ordinal_encoding

    def get_feature_values(self):
        return self._encoder_feature_values.copy()

    def _training_discrete_conversion(self, training_array):
        # conversion from discrete values -> numeric values
        for feature in list(training_array):
            unique_values = list(np.unique(training_array[feature].astype(str)))
            try:
                unique_values.remove('nan')
            except ValueError:
                pass
            if (len(unique_values) <= self._category_threshold or
                    not all([check_numeric(value) for value in unique_values])):
                if (not CategoricalEncoder._IGNORE_NUMERIC and
                        all([check_numeric(value) for value in unique_values])):
                    self._finite_numeric_features.append(feature)
                self._categorical_features.append(feature)
                print("Training Array: ", training_array[feature])
                self._feature_original_types[feature] = type(training_array[feature].values[0])
                self._encoder_key[feature] = list(np.array(unique_values).astype(self._feature_original_types[feature]))

    def _make_encoder_key(self, training_data=None, initial_key=None):
        if initial_key is None or (not initial_key and training_data is not None):
            self._categorical_features = []  # indicators of which columns are categorical
            self._feature_original_types = {}  # indicates the original type of all categorical values
            self._encoder_key = dict()
            self._finite_numeric_features = []
            self._training_discrete_conversion(training_data)
        else:
            self._feature_original_types = {}
            self._encoder_key = initial_key
            self._categorical_features = list(initial_key.keys())
            self._finite_numeric_features = []
            for key in initial_key.keys():
                self._feature_original_types[key] = type(initial_key[key][0])
                if (not CategoricalEncoder._IGNORE_NUMERIC and
                        all([check_numeric(initial_key[key][i]) for i in range(len(initial_key[key]))])):
                    self._finite_numeric_features.append(key)

    def _make_feature_order(self, data_features):
        new_feature_order = data_features.copy()
        if not self._ordinal:
            for feature in self._encoder_key.keys():
                if feature not in self._finite_numeric_features:
                    for value in self._encoder_key[feature]:
                        new_feature_order.insert(new_feature_order.index(feature), feature + "=" + str(value))
                    new_feature_order.remove(feature)
        self._encoder_feature_values = new_feature_order

    def _make_bounds(self, training_data=None):
        if training_data is not None and self._find_non_category_bounds:
            self._non_category_bounds = []
            categorical_features = self._encoder_key.keys()
            for feature in self._original_feature_order:
                bounds = None
                if feature not in categorical_features:
                    bounds = (np.nanmin(training_data[feature]),
                              np.nanmax(training_data[feature]))
                self._non_category_bounds.append(bounds)

    def fit(self, training_data=None, initial_key=None, data_features=None, data_means=None):
        if data_features is None and training_data is not None:
            data_features = training_data.columns
        self._make_encoder_key(training_data=training_data, initial_key=initial_key)
        self._make_feature_order(data_features)
        self._make_bounds(training_data=training_data)

    def transform(self, data):
        self._original_feature_order = list(data) if len(data.shape) > 1 else list(data.index)
        enc_data = data.copy()
        for feature in self._encoder_key.keys():
            if feature not in self._finite_numeric_features:
                feature_flag = False
                for value in self._encoder_key[feature]:
                    str_value = str(value)
                    # two layers the first is a dictionary containing the features, the second is values that feature takes
                    if not self._ordinal:
                        if len(enc_data.shape) > 1:
                            enc_data[feature + "=" + str_value] = 0
                            enc_data.loc[(enc_data[feature]).astype(str) == str_value, feature + "=" + str_value] = 1
                        else:
                            enc_data[feature + "=" + str_value] = 0 if enc_data[feature] != str_value else 1
                    else:
                        if len(enc_data.shape) > 1:
                            enc_search = np.where(data[feature].astype(str) == str_value)[0]
                            enc_replace = np.argwhere(np.array(self._encoder_key[feature]).astype(str) == str_value)[0][0]
                            if not feature_flag:
                                enc_data[feature] = 0
                                feature_flag = True
                        else:
                            if enc_data[feature] == str_value:
                                enc_data[feature] = (
                                    np.argwhere(np.array(self._encoder_key[feature]) == value)[0][0])
                if not self._ordinal:
                    if len(enc_data.shape) > 1:
                        enc_data.drop([feature], axis=1, inplace=True)
                    else:
                        enc_data.drop([feature], axis=0, inplace=True)

        return enc_data[self._encoder_feature_values]

    def fit_transform(self, training_data=None, initial_key=None, data_features=None):
        self.fit(training_data=training_data, initial_key=initial_key, data_features=data_features)
        if training_data is not None:
            return self.transform(training_data)

    def inverse_transform(self, enc_data):
        if isinstance(enc_data, np.ndarray):
            enc_data = pd.DataFrame(enc_data, columns=self._encoder_feature_values)
        data = enc_data.copy()
        for feature in self._encoder_key.keys():
            if feature not in self._finite_numeric_features:
                feature_columns = [feature + "=" + str(value) for value in self._encoder_key[feature]]
                data[feature] = self._encoder_key[feature][0] if not self._ordinal else 0
                for i in range(enc_data.shape[0]):
                    if not self._ordinal:
                        data[feature][i] = self._encoder_key[feature][np.argmax(np.array(data.iloc[i][feature_columns]))]
                    else:
                        data[feature][i] = self._encoder_key[feature][np.argmin(
                            np.abs(np.array([i for i in range(len(self._encoder_key[feature]))]) - int(enc_data[feature][i])))]

                data[feature] = data[feature].astype(self._feature_original_types[feature])
            else:
                for i in range(enc_data.shape[0]):
                    data[feature][i] = self._encoder_key[feature][np.argmin(
                        np.abs(np.array(self._encoder_key[feature]) - data[feature][i]))]
        return data[self._original_feature_order]

    def rescale_categorical(self, data, means=None, current_category_bias=0):
        rescaled_data = data.copy()
        for feature in self._encoder_key.keys():
            if not self._ordinal and feature not in self._finite_numeric_features:
                n_values = len(self._encoder_key[feature]) + 1
                for value in self._encoder_key[feature]:
                    if means is None:
                        rescaled_data[feature + "=" + str(value)] = (
                            CategoricalEncoder.DEFAULT_CAT_BIAS['current'](n_values)) \
                            if rescaled_data[feature + "=" + str(value)] == 1 \
                            else CategoricalEncoder.DEFAULT_CAT_BIAS['other'](n_values)
                    else:
                        feature_position = np.argwhere(np.array(self._encoder_feature_values) ==
                                                       (feature + "=" + str(value)))[0][0]
                        if current_category_bias == "avg_means" and rescaled_data[feature + "=" + str(value)] == 1:
                            cat_avg = 0
                            cat_count = 1
                            for value2 in self._encoder_key[feature]:
                                if value2 != value:
                                    feature_position2 = np.argwhere(np.array(self._encoder_feature_values) ==
                                                                   (feature + "=" + str(value2)))[0][0]
                                    cat_avg += means[feature_position2]
                                    cat_count += 1
                            cat_avg /= cat_count
                            rescaled_data[feature + "=" + str(value)] = means[feature_position] + cat_avg
                        else:
                            rescaled_data[feature + "=" + str(value)] = means[feature_position] + current_category_bias \
                                if rescaled_data[feature + "=" + str(value)] == 1 else means[feature_position]

        return rescaled_data

    def get_encoder_key(self):
        return self._encoder_key.copy()

    def get_categorical_features(self):
        return self._categorical_features.copy()

    def get_categorical_indices(self):
        return [i for i in range(len(self._encoder_feature_values)) if "=" in self._encoder_feature_values[i]]

    def get_bounds(self):
        return self._non_category_bounds.copy()


class SigDirectEncoder(CategoricalEncoder):
    """
    Purpose: CategoricalEncoder that can also handle rule formats used by SigDirect/SigD2.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _count_rule_size(self, rule):
        return sum(rule)

    def _count_applicable(self, enc_row, rule):
        return sum(enc_row & rule)

    def check_applicable(self, data_row, rule):
        # check that the current data is applicable to the current rule
        enc_rule = rule[0]
        enc_row = self.transform(data_row)
        return self._count_rule_size(rule) == self._count_applicable(enc_row, enc_rule)

    def decode_rule(self, rule, as_string=True):
        # decode a rule into the string representation 1 < a < 2 -> y
        # if not as_string return open ranges and values represented by it
        pass

    def get_rule_importance(self, rule):
        # get the feature importance that a rule applies (collectively)
        pass

    def get_rule_features(self, rule):
        # get the features that are used by the rule
        enc_rule = rule[0]
        return self._encoder_feature_values[enc_rule]

    def get_rule_center_points(self, rule, restricted_centers=None):
        # get the center points for all values in a rule
        # restricted_centers tells how values should be able to change
        enc_rule = rule[0]
        next_enc_rule = list()
        prev_feature = None
        n_enc_feats = len(self._encoder_feature_values)
        for i in range(n_enc_feats):
            feature = self._encoder_feature_values[i]
            if (feature != prev_feature or
                    i == (n_enc_feats-1) or
                    "=" in feature):
                # what to do on either end??
                next_enc_rule.append(enc_rule[i])  # get current value if end point
            else:
                set_on = 1 if enc_rule[i-1] == 1 else 0  # get next rule
                next_enc_rule.append(set_on)

        inv_rule = self.inverse_transform(enc_rule)
        inv_next_rule = self.inverse_transform(next_enc_rule)

        # using the next value if it's there get center point
        for i in range(len(inv_rule)):
            if check_numeric(inv_rule[i]):
                inv_rule[i] = (inv_rule[i] + inv_next_rule[i]) / 2

        return inv_rule

    def get_rule_distance_from_data(self, data_row, rule, distance_metrics):
        # take a data_row and rule and get the distance of the rule from the data as a number
        pass

    def get_distance_from_data(self, data_row1, data_row2, distance_metrics):
        # show the distance between two rows of data
        pass

    def get_discrete_alternative(self, data_row, feature, restricted_discrete=None):
        # get alternative value that feature of data_row could take on not in the restricted list
        #  if there are none that do not appear in the list then use the first item in the list
        if restricted_discrete is None:
            restricted_discrete = list()
