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

    def _training_discrete_conversion(self, training_array):
        # conversion from discrete values -> numeric values
        for feature in list(training_array):
            #print(feature)
            #print(training_array[feature])
            unique_values = list(np.unique(training_array[feature].astype(str)))
            try:
                unique_values.remove('nan')
            except ValueError:
                pass
            if not all([check_numeric(value) for value in unique_values]):
                #print("IAIN UNIQUES ", unique_values)
                #print(all([check_numeric(value) for value in unique_values]))
                #print([check_numeric(value) for value in unique_values])
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
        #print("KEYS FROM DISCRETIZER: ", self._encoder_key)

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
            #print(self._encoder_key)

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
            data_features = list(training_data)
        self._make_encoder_key(training_data=training_data, initial_key=initial_key)
        self._make_feature_order(data_features)
        self._make_bounds(training_data=training_data)

    def transform(self, data):
        #print("TRANSFORM DATA: ", data)
        self._original_feature_order = list(data) if len(data.shape) > 1 else list(data.index)
        enc_data = data.copy()
        #new_feature_order = list(data) if len(data.shape) > 1 else list(data.index)
        for feature in self._encoder_key.keys():
            if feature not in self._finite_numeric_features:
                feature_flag = False
                for value in self._encoder_key[feature]:
                    str_value = str(value)
                    # two layers the first is a dictionary containing the features, the second is values that feature takes
                    #print(enc_data.shape)
                    if not self._ordinal:
                        if len(enc_data.shape) > 1:
                            enc_data[feature + "=" + str_value] = 0
                            enc_data[feature + "=" + str_value].loc[(enc_data[feature]).astype(str) == str_value] = 1
                        else:
                            enc_data[feature + "=" + str_value] = 0 if enc_data[feature] != str_value else 1
                        #print(new_feature_order)
                        #new_feature_order.insert(new_feature_order.index(feature), feature + "=" + str_value)
                    else:
                        #print(np.where(enc_data[feature] == value))
                        #print(np.argwhere(np.array(self._encoder_key[feature]) == value))
                        #print(self._encoder_key[feature])
                        #print(value)
                        #print(np.where(enc_data[feature] == value)[0])
                        #print(enc_data[feature])
                        if len(enc_data.shape) > 1:
                            enc_search = np.where(data[feature].astype(str) == str_value)[0]
                            enc_replace = np.argwhere(np.array(self._encoder_key[feature]).astype(str) == str_value)[0][0]

                            #print("ORDINAL_VALUE: ", value)
                            #print("ORDINAL SEARCH: ", enc_search)
                            #print("ORDINAL REPLACE: ", enc_replace)
                            if not feature_flag:
                                enc_data[feature] = 0
                                feature_flag = True
                            #enc_data[feature][enc_search] = enc_replace.copy()
                        else:
                            if enc_data[feature] == str_value:
                                enc_data[feature] = (
                                    np.argwhere(np.array(self._encoder_key[feature]) == value)[0][0])
                if not self._ordinal:
                    if len(enc_data.shape) > 1:
                        enc_data.drop([feature], axis=1, inplace=True)
                    else:
                        enc_data.drop([feature], axis=0, inplace=True)
                    #new_feature_order.remove(feature)

        return enc_data[self._encoder_feature_values]

    def fit_transform(self, training_data=None, initial_key=None, data_features=None):
        self.fit(training_data=training_data, initial_key=initial_key, data_features=data_features)
        if training_data is not None:
            return self.transform(training_data)

    def inverse_transform(self, enc_data):
        #print(type(enc_data))
        #print("INVERSE CALL: ", enc_data)
        if isinstance(enc_data, np.ndarray):
            enc_data = pd.DataFrame(enc_data, columns=self._encoder_feature_values)
        data = enc_data.copy()
        for feature in self._encoder_key.keys():
            if feature not in self._finite_numeric_features:
                feature_columns = [feature + "=" + str(value) for value in self._encoder_key[feature]]
                data[feature] = self._encoder_key[feature][0] if not self._ordinal else 0
                for i in range(enc_data.shape[0]):
                    #print(np.array(data.iloc[i][feature_columns]))
                    #print(np.argmax(np.array(data.iloc[i][feature_columns])))
                    #print(self._encoder_key[feature])
                    #print(data.iloc[0:10][feature_columns])

                    if not self._ordinal:
                        data[feature][i] = self._encoder_key[feature][np.argmax(np.array(data.iloc[i][feature_columns]))]
                    else:
                        #print(data[feature])
                        #print(np.array([i for i in range(len(self._encoder_key[feature]))]))
                        data[feature][i] = self._encoder_key[feature][np.argmin(
                            np.abs(np.array([i for i in range(len(self._encoder_key[feature]))]) - int(enc_data[feature][i])))]
                    #print(np.argmax(np.array(data.iloc[i][feature_columns])))
                    #print(data.iloc[i][feature])
                data[feature] = data[feature].astype(self._feature_original_types[feature])
            else:
                for i in range(enc_data.shape[0]):
                    data[feature][i] = self._encoder_key[feature][np.argmin(
                        np.abs(np.array(self._encoder_key[feature]) - data[feature][i]))]
        return data[self._original_feature_order]

    def rescale_categorical(self, data, means=None, current_category_bias=0):
        #print(self._encoder_feature_values)
        #print(means)
        #print(current_category_bias)
        #assert False
        rescaled_data = data.copy()
        for feature in self._encoder_key.keys():
            if not self._ordinal and feature not in self._finite_numeric_features:
                n_values = len(self._encoder_key[feature]) + 1
                for value in self._encoder_key[feature]:
                    # TODO: IAIN NEW STUFF
                    # TODO: find a good criterion for the base value of categorical
                    # TODO: check if starting from a biased value similar to the mean is better??
                    #  TODO: it is possible that this is a convex function wrt the accuracy on both training and unknown
                    #  TODO: compare results with the numbered 1, 2, 3, 4 categories
                    #  TODO: add analysis of beginning starting points that can be used in paper or thesis
                    if means is None:
                        rescaled_data[feature + "=" + str(value)] = (
                            CategoricalEncoder.DEFAULT_CAT_BIAS['current'](n_values)) \
                            if rescaled_data[feature + "=" + str(value)] == 1 \
                            else CategoricalEncoder.DEFAULT_CAT_BIAS['other'](n_values)
                    else:
                        #print(self._encoder_feature_values)
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
        #print(rescaled_data)
        return rescaled_data

    def get_encoder_key(self):
        return self._encoder_key.copy()

    def get_categorical_features(self):
        return self._categorical_features.copy()

    def get_categorical_indices(self):
        return [i for i in range(len(self._encoder_feature_values)) if "=" in self._encoder_feature_values[i]]

    def get_bounds(self):
        return self._non_category_bounds.copy()
