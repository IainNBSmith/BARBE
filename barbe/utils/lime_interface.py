"""
All CODE USED FROM LIME () AND VAE-LIME () REPOSITORIES
TO REPEAT TESTS LOCAL NAMES MAY NEED TO BE CHANGED TO YOUR LOCAL OR INSTALLED
 DIRECTORY PATHS (LOOK FOR COMMENTS # !!LOCAL!! #)
- uses: lime_base lime_tabular (from any LIME-derived code)
- names of functions are the same in some implementations load with
   different names using 'import ... as ...'
- code changed locally for utility default operating remains exactly the same
   (detailed changes follow)

Purpose: Take LIME-derived code and reconfigure required functions to have a
 BARBE format. Used in experiments when calling for perturbations and surrogate
 predictions that are not handled in LIME-derived code.

Modified functions for LIME-derived code include:
    - lime_base.explain_instance_with_data: save local models as ...
       self.local_models -> now contains local model for each class
    - lime_tabular.__data_inverse: no changes, used for the sake of calling
       (it is a hidden function __ so we cannot trust it will always be the same)
    - lime_tabular.explain_instance: saves perturbed data (self.perturbed_data) and
       optionally calls BARBE perturber instead to generate perturbations

Added functions to LIME-derived code include:
    - lime_base.predict: make prediction for f_k(.) a predictor (for different
       evaluations) f(x) = label[argmax_k(f_k(x))]
                    f_k(x) = Ridge(x) <- trained to classify the kth class
    - lime_tabular.__barbe_data_inverse: produce perturbations using BARBE techniques

"""
import pandas as pd

import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances
from numpy.random import RandomState
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_base import LimeBase
from barbe.perturber import BarbePerturber, ClassBalancedPerturber
from barbe.utils.evaluation_measures import *
from barbe.utils.bbmodel_interface import *
from barbe.discretizer import CategoricalEncoder
import scipy as sp
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import copy
from lime.lime_tabular import TableDomainMapper
from lime import explanation
import warnings

from slime.lime_tabular import LimeTabularExplainer as SLimeTabularExplainer
from slime.lime_base import LimeBase as SLimeBase

# Load requirements from VAE-LIME named differently
from VAELIME.lime2.lime_tabular import LimeTabularExplainer as VAELimeTabularExplainer  # !!LOCAL!! #
from VAELIME.lime2.lime_base import LimeBase as VAELimeBase  # !!LOCAL!! #
from VAELIME.lime2.lime_tabular import TableDomainMapper as VAETableDomainMapper  # !!LOCAL!! #
from VAELIME.lime2 import explanation as vae_explanation  # !!LOCAL!! #
from VAELIME.Generators.VAE import VAE  # !!LOCAL!! #
from VAELIME.Generators.DropoutVAE import DropoutVAE  # !!LOCAL!! #
# TODO/DONE: add interface of VAELime
# TODO: compress code repetitions via object oriented programming


# **************** ORIGINAL LIME **************** #
class LimeNewBase(LimeBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_model = {}
        self.used_features = {}

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
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
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)
        #print("LIME: ", label, " pred score - ", prediction_score)

        self.used_features[label] = used_features

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        self.local_model[label] = easy_model
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred, )
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)


class LimeNewPert(LimeTabularExplainer):

    def __init__(self, training_data=None, training_labels=None, **kwargs):
        self.encoder = CategoricalEncoder(ordinal_encoding=False)
        self.diff_encoder = CategoricalEncoder(ordinal_encoding=False)
        self.training_data = training_data.copy()
        self.diff_encoder.fit(training_data.copy())
        training_data = self.encoder.fit_transform(training_data=training_data)
        super().__init__(training_data=training_data, training_labels=training_labels, **kwargs)
        self.base = LimeNewBase(kernel_fn=self.base.kernel_fn,
                                verbose=self.base.verbose,
                                random_state=self.base.random_state)
        self.unique_labels = np.unique(training_labels)
        self.perturbed_data = None

    def predict(self, X):
        if self.encoder is not None:
            X = self.encoder.transform(X).to_numpy()
            if len(X.shape) < 2:
                X = X.reshape(-1, 1)
        X_np = self.scale_data(X)
        model_predictions = None
        for c_label in self.base.used_features.keys():
            c_predictions = self.base.local_model[c_label].predict(X_np[:, self.base.used_features[c_label]])
            if model_predictions is None:
                model_predictions = c_predictions
            else:
                model_predictions = np.vstack([model_predictions, c_predictions])
        model_predictions = model_predictions.T
        if X_np.shape[0] == 1 and False:
            print(model_predictions)
            print(np.argmax(model_predictions, axis=1))
            print(self.unique_labels[np.argmax(model_predictions, axis=1)].astype(str))
            assert False
        return self.unique_labels[np.argmax(model_predictions, axis=1)].astype(str)

    def scale_data(self, X):
        return (X - self.scaler.mean_) / self.scaler.scale_

    def __data_inverse(self,
                       data_row,
                       num_samples,
                       sampling_method):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
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

            if sampling_method == 'gaussian':
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1] * num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                #print(data.shape)
                #print((data * scale).shape)
                #print(instance_sample.shape)
                data = np.array(data * scale) + np.array(instance_sample)
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
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.encoder is not None:
            inverse = self.encoder.inverse_transform(inverse).to_numpy()
        # inverse[0] = self.encoder.inverse_transform(data_row).to_numpy()
        # rough cut categories
        data = self.encoder.transform(self.encoder.inverse_transform(data)).to_numpy()
        return data, inverse

    def __barbe_data_inverse(self,
                             data_row,
                             num_samples,
                             barbe_pert_model='normal',
                             barbe_dev_scaling=1):
        # TODO: NOTE - magnify the distribution size to get results similar to them on lime2, which is not good...
        self.training_data = pd.DataFrame(self.training_data, columns=self.feature_names)
        bp = ClassBalancedPerturber(training_data=self.training_data,
                                    perturbation_type=barbe_pert_model,
                                    dev_scaling_factor=barbe_dev_scaling,
                                    standardized_categorical_variance=True,
                                    use_mean_categorical_odds=True,
                                    balance_mode='curr-other')
        data = bp.produce_perturbation(num_samples, data_row=data_row)
        inverse = data.copy()
        if self.encoder is not None:
            data = self.encoder.transform(data)
        return data.to_numpy(), inverse

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian',
                         barbe_mode=False,
                         barbe_pert_model='normal',
                         barbe_dev_scaling=1):
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
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        if barbe_mode:
            data, inverse = self.__barbe_data_inverse(data_row, num_samples,
                                                      barbe_pert_model=barbe_pert_model,
                                                      barbe_dev_scaling=barbe_dev_scaling)
            if self.encoder is not None:
                data_row = self.encoder.transform(data_row)
        else:
            og_data_row = data_row.copy()
            data_row = self.encoder.transform(data_row)
            data, inverse = self.__data_inverse(data_row, num_samples, sampling_method)
            inverse[0,:] = og_data_row
            inverse = pd.DataFrame(inverse, columns=self.encoder._original_feature_order)
            #print("INV: ", inverse)
            #inverse = self.encoder.inverse_transform(inverse)
        self.perturbed_data = inverse.copy()
        if sp.sparse.issparse(data):
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

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
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
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
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
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=self.encoder._encoder_feature_values.copy(),
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
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
        ret_exp.score = {}
        ret_exp.local_pred = {}
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score, weights=None, original_data=None):
        wrapped_comparison = BlackBoxWrapper(comparison_model)
        if self.encoder is not None:
            discretize_call = self.diff_encoder.transform
        else:
            discretize_call = lambda x: x

        if weights is not None and weights in 'euclidean':
            if comparison_data is None:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(self.perturbed_data))
            else:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(comparison_data).to_numpy())
        elif weights is not None and weights in 'nearest-neighbors':
            #print("OG DATA: ", discretize_call(self.perturbed_data))
            #print("Model Prediction: ", self.predict(discretize_call(self.perturbed_data)))
            #print("BB Prediction: ", wrapped_comparison.predict(self.perturbed_data))
            if comparison_data is None:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(self.perturbed_data))
            else:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(comparison_data).to_numpy())
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        if (comparison_model is None) and (comparison_data is None):
            #self.perturbed_data = pd.DataFrame(self.perturbed_data, columns=[str(i) for i in range(4)])
            #print('Perturbed Data: ', self.perturbed_data)
            #print('Perturbed Result: ', wrapped_comparison.predict(self.perturbed_data))
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        #elif (comparison_model is None) and (comparison_data is not None):
        #    return comparison_method(self._blackbox_classification['perturbed'],
        #                             self._surrogate_classification['perturbed'],
        #                             sample_weight=weights)

        elif (comparison_model is not None) and (comparison_data is None):
            #print("BBMODEL: ", wrapped_comparison.predict(self.perturbed_data))
            #print("LIME: ", self.predict(self.perturbed_data))
            #print("WEIGHTS: ", weights)
            #print("COMPARISON: ", comparison_method(wrapped_comparison.predict(self.perturbed_data),
            #                         self.predict(self.perturbed_data),
            #                         sample_weight=weights))
            #self.perturbed_data = pd.DataFrame(self.perturbed_data, columns=[str(i) for i in range(4)])
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        elif (comparison_model is not None) and (comparison_data is not None):
            return comparison_method(wrapped_comparison.predict(comparison_data),
                                     self.predict(comparison_data),
                                     sample_weight=weights)


# **************** VAE-LIME **************** #
# TODO: NOTE: it seems VAE-LIME gets large proportions of all three classes (iris)
#  this still does not work well for Ridge and most notably it is usually not
#  done since there is a large GAP in certain classes (why should or shouldn't
#  we use this???)
class VAELimeNewBase(VAELimeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_model = {}
        self.used_features = {}

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
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
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        self.used_features[label] = used_features

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        self.local_model[label] = easy_model

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)


class VAELimeNewPert(VAELimeTabularExplainer):

    def __init__(self, training_data=None, training_labels=None, **kwargs):
        self.encoder = CategoricalEncoder(ordinal_encoding=False)
        self.training_data = training_data.copy()
        self.encoder.fit(training_data.copy())
        training_data = self.encoder.fit_transform(training_data=training_data)
        super().__init__(training_data=training_data, training_labels=training_labels,
                         generator='VAE', generator_specs={'original_dim': training_data.shape[1],
                                                           'input_shape': (100, training_data.shape[1]),
                                                           'intermediate_dim': 100,  # 100 for larger data
                                                           'latent_dim': 4,  # 4
                                                           'epochs': 100},
                         dummies=[],
                         **kwargs)
        self.base = VAELimeNewBase(kernel_fn=self.base.kernel_fn,
                                   verbose=self.base.verbose,
                                   random_state=self.base.random_state)
        self.unique_labels = np.unique(training_labels)
        self.perturbed_data = None

    def scale_data(self, X):
        return (X - self.scaler.mean_) / self.scaler.scale_

    def predict(self, X):
        if self.encoder is not None:
            X = self.encoder.transform(X).to_numpy()
            if len(X.shape) < 2:
                X = X.reshape(-1, 1)
        X_np = self.scale_data(X)
        #X_np = X
        model_predictions = None
        for c_label in self.base.used_features.keys():
            c_predictions = self.base.local_model[c_label].predict(X_np[:, self.base.used_features[c_label]])
            if model_predictions is None:
                model_predictions = c_predictions
            else:
                model_predictions = np.vstack([model_predictions, c_predictions])
        model_predictions = model_predictions.T
        return self.unique_labels[np.argmax(model_predictions, axis=1)].astype(str)

    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
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
            # If we use perturbations to generate new samples, we have standard scaler and need mean and variance of the data
            if self.generator is None:
                scale = self.scaler.scale_
                mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]
            # Generate samples using the given generator
            if self.generator == "RBF":
                # With RBF and Forest, we load data which were generated in R
                if self.generator_specs["experiment"] == "Compas":
                    df = pd.read_csv("..\Data\compas_RBF.csv")
                elif self.generator_specs["experiment"] == "German":
                    df = pd.read_csv("..\Data\german_RBF.csv")
                else:
                    df = pd.read_csv("..\Data\cc_RBF.csv")
                # There are no nominal features in CC dataset
                if self.generator_specs["experiment"] != "CC":
                    df = pd.get_dummies(df)
                    df = df[self.feature_names]
                inverse = df.values
                inverse[0, :] = data_row
                data = inverse.copy()
                for feature in categorical_features:
                    data[:, feature] = (inverse[:, feature] == data_row[feature]).astype(int)
                return data, inverse
            if self.generator == "Forest":
                if self.generator_specs["experiment"] == "Compas":
                    df = pd.read_csv("..\Data\compas_forest.csv")
                elif self.generator_specs["experiment"] == "German":
                    df = pd.read_csv("..\Data\german_forest.csv")
                else:
                    df = pd.read_csv("..\Data\cc_forest.csv")
                if self.generator_specs["experiment"] != "CC":
                    df = pd.get_dummies(df)
                    df = df[self.feature_names]
                inverse = df.values
                inverse[0, :] = data_row
                data = inverse.copy()
                for feature in categorical_features:
                    data[:, feature] = (inverse[:, feature] == data_row[feature]).astype(int)
                return data, inverse
            # Perturbations
            if self.generator is None:
                data = self.random_state.normal(
                    0, 1, num_samples * num_cols).reshape(
                    num_samples, num_cols)
                if self.sample_around_instance:
                    data = data * scale + instance_sample
                else:
                    data = data * scale + mean
                # With VAE and DropoutVAE we generate new data in vicinity of data_row
            elif isinstance(self.generator, VAE):
                reshaped = data_row.to_numpy().reshape(1, -1)  # IAIN changed
                scaled = self.generator_scaler.transform(reshaped)
                encoded = self.generator.encoder.predict(scaled)
                encoded = np.asarray(encoded)

                results = []
                latent_gen = []
                for _ in range(num_samples):
                    epsilon = np.random.normal(0., 1., encoded.shape[2])
                    latent_gen.extend([encoded[0, 0, :] + np.exp(encoded[1, 0, :] * 0.5) * epsilon])
                latent_gen = np.asarray(latent_gen)
                results.append(self.generator.generate(latent_gen))

                results = np.asarray(results)
                results = np.reshape(results, (-1, len(data_row)))

                data = self.generator_scaler.inverse_transform(results)

            elif isinstance(self.generator, DropoutVAE):
                reshaped = data_row.reshape(1, -1)
                scaled = self.generator_scaler.transform(reshaped)
                scaled = np.reshape(scaled, (-1, len(data_row)))
                encoded = self.generator.mean_predict(scaled, nums=num_samples)

                results = encoded
                results = results.reshape(num_samples * results.shape[2], len(data_row))
                data = self.generator_scaler.inverse_transform(results)

                # Round up integer attributes
                data[:, self.integer_attributes] = (np.around(data[:, self.integer_attributes])).astype(int)

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
        inverse = data.copy()
        if self.generator is None:
            for column in categorical_features:
                values = self.feature_values[column]
                freqs = self.feature_frequencies[column]
                inverse_column = self.random_state.choice(values, size=num_samples,
                                                          replace=True, p=freqs)
                binary_column = (inverse_column == first_row[column]).astype(int)
                binary_column[0] = 1
                inverse_column[0] = data[0, column]
                data[:, column] = binary_column
                inverse[:, column] = inverse_column
        # We assume categorical features are binary encoded
        #else:
            #for feature in self.dummies:
            #    column = data[:, feature]
            #    binary = np.zeros(column.shape)
            #    # We check for binary features with only 2 possible values
            #    if len(feature) == 1:
            #        binary = (column > 0.5).astype(int)
            #        # Put ones in data, where the value of chosen feature is same as in data_row
            #        data[:, feature] = (binary == first_row[feature]).astype(int)
            #    else:
            #        # Delegate 1 to the dummy_variable with the highest value
            #        ones = column.argmax(axis=1)
            #        for i, idx in enumerate(ones):
            #            binary[i, idx] = 1
            #        # Put ones in data, where the value of chosen feature is same as in data_row
            #        for i, idx in enumerate(feature):
            #            data[:, idx] = (binary[:, i] == first_row[idx]).astype(int)
            #    inverse[:, feature] = binary
        if self.encoder is not None:
            # print("CURR: ", inverse)
            inverse = self.encoder.inverse_transform(inverse)
            data = self.encoder.transform(self.encoder.inverse_transform(data))
        inverse[0] = data_row
        return data, inverse

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         # dummies for testing
                         barbe_mode=None,
                         barbe_pert_model=None,
                         barbe_dev_scaling=None):
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
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        # og_data_row = data_row.copy()
        data_row = self.encoder.transform(data_row)
        data, inverse = self.__data_inverse(data_row, num_samples)
        # print("CURR DATA: ", data)
        data = data.to_numpy()
        # inverse[0, :] = og_data_row
        inverse = pd.DataFrame(inverse, columns=self.encoder._original_feature_order)
        self.perturbed_data = inverse.copy()
        if sp.sparse.issparse(data):
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

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
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
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
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
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=self.encoder._encoder_feature_values.copy(),
                                          feature_indexes=feature_indexes)
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
        ret_exp.score = {}
        ret_exp.local_pred = {}
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score, weights=None, original_data=None):
        wrapped_comparison = BlackBoxWrapper(comparison_model)
        if self.encoder is not None:
            discretize_call = self.encoder.transform
        else:
            discretize_call = lambda x: x
        if weights is not None and weights in 'euclidean':
            if comparison_data is None:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(self.perturbed_data))
            else:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(comparison_data).to_numpy())
        elif weights is not None and weights in 'nearest-neighbors':
            if comparison_data is None:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(self.perturbed_data))
            else:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(comparison_data).to_numpy())
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        if (comparison_model is None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        #elif (comparison_model is None) and (comparison_data is not None):
        #    return comparison_method(self._blackbox_classification['perturbed'],
        #                             self._surrogate_classification['perturbed'],
        #                             sample_weight=weights)

        elif (comparison_model is not None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        elif (comparison_model is not None) and (comparison_data is not None):
            return comparison_method(wrapped_comparison.predict(comparison_data),
                                     self.predict(comparison_data),
                                     sample_weight=weights)


# **************** S-LIME **************** #
class SLimeNewBase(SLimeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_model = {}
        self.used_features = {}

    def testing_explain_instance_with_data(self,
                                           neighborhood_data,
                                           neighborhood_labels,
                                           distances,
                                           label,
                                           num_features,
                                           feature_selection='lasso_path',
                                           model_regressor=None,
                                           alpha=0.05):
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
        # used_features, test_results
        print("LARS")
        print(weights.T)
        print(labels_column.T)
        print(neighborhood_data)
        print(num_features)
        print(alpha)
        used_features, test_results = self.feature_selection(neighborhood_data,
                                               labels_column.T,
                                               weights.T,
                                               num_features,
                                               feature_selection,
                                               testing=True,
                                               alpha=alpha)
        #print("YOU CALLED ")
        #print(aa)
        #assert False
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        self.used_features[label] = used_features

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        self.local_model[label] = easy_model

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred, used_features, test_results)


class SLimeNewPert(SLimeTabularExplainer):

    def __init__(self, training_data=None, training_labels=None, **kwargs):
        self.encoder = CategoricalEncoder(ordinal_encoding=False)
        self.training_data = training_data.copy()
        self.encoder.fit(training_data.copy())
        training_data = self.encoder.fit_transform(training_data=training_data)
        super().__init__(training_data=training_data, training_labels=training_labels, **kwargs)
        self.base = SLimeNewBase(kernel_fn=self.base.kernel_fn,
                                 verbose=self.base.verbose,
                                 random_state=self.base.random_state)
        self.unique_labels = np.unique(training_labels)
        self.perturbed_data = None

    def scale_data(self, X):
        return (X - self.scaler.mean_) / self.scaler.scale_

    def predict(self, X):
        if self.encoder is not None:
            X = self.encoder.transform(X).to_numpy()
            if len(X.shape) < 2:
                X = X.reshape(-1, 1)
        X_np = self.scale_data(X)
        #X_np = X
        model_predictions = None
        for c_label in self.base.used_features.keys():
            c_predictions = self.base.local_model[c_label].predict(X_np[:, self.base.used_features[c_label]])
            if model_predictions is None:
                model_predictions = c_predictions
            else:
                model_predictions = np.vstack([model_predictions, c_predictions])
        model_predictions = model_predictions.T
        return self.unique_labels[np.argmax(model_predictions, axis=1)].astype(str)

    def slime(self,
              data_row,
              predict_fn,
              labels=(1,),
              top_labels=None,
              num_features=10,
              num_samples=1000,
              distance_metric='euclidean',
              model_regressor=None,
              sampling_method='gaussian',
              n_max=10000,
              alpha=0.05,
              tol=1e-3):
        """Generates explanations for a prediction with S-LIME.

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
            num_samples: size of the neighborhood to learn the linear model as a start
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.
            n_max: maximum number of sythetic samples to generate.
            alpha: significance level of hypothesis testing.
            tol: tolerence level of hypothesis testing.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        itert = 0
        exit_loop = False
        while not exit_loop and itert < 10:
            ret_exp, test_result = self.testing_explain_instance(data_row=data_row,
                                                                 predict_fn=predict_fn,
                                                                 labels=labels,
                                                                 top_labels=top_labels,
                                                                 num_features=num_features,
                                                                 num_samples=num_samples,
                                                                 distance_metric=distance_metric,
                                                                 model_regressor=model_regressor,
                                                                 sampling_method=sampling_method,
                                                                 alpha=alpha)
            itert += 1
            where_stop = num_features
            flag = False
            for k in range(1, num_features + 1):
                if (not flag) and test_result[k][0] < -tol:
                    flag = True
                    where_stop = k
            # fix unending loops
            if num_samples != n_max and n_max > int(test_result[where_stop][1]) > 2 * num_samples:
                num_samples = min(int(test_result[where_stop][1]), 2 * num_samples)
                if num_samples > n_max:
                    num_samples = n_max
            elif n_max <= int(test_result[where_stop][1]):
                num_samples = n_max
            else:
                exit_loop = True

        return ret_exp

    def __data_inverse(self,
                       data_row,
                       num_samples,
                       sampling_method):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
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

            if sampling_method == 'gaussian':
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1] * num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                # print(data.shape)
                # print((data * scale).shape)
                # print(instance_sample.shape)
                data = np.array(data * scale) + np.array(instance_sample)
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
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.encoder is not None:
            inverse = self.encoder.inverse_transform(inverse).to_numpy()
        # inverse[0] = self.encoder.inverse_transform(data_row).to_numpy()
        # rough cut categories
        data = self.encoder.transform(self.encoder.inverse_transform(data)).to_numpy()
        return data, inverse

    def testing_explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         # dummies for testing
                         sampling_method='gaussian',
                         alpha=0.05,
                         barbe_mode=None,
                         barbe_pert_model=None,
                         barbe_dev_scaling=None):
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
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        # og_data_row = data_row.copy()
        data_row = self.encoder.transform(data_row)
        data, inverse = self.__data_inverse(data_row, num_samples, 'gaussian')
        # print("CURR DATA: ", data)
        #data = data.to_numpy()
        # inverse[0, :] = og_data_row
        inverse = pd.DataFrame(inverse, columns=self.encoder._original_feature_order)
        self.perturbed_data = inverse.copy()
        if sp.sparse.issparse(data):
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

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
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
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
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
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=self.encoder._encoder_feature_values.copy(),
                                          feature_indexes=feature_indexes)
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
        ret_exp.score = {}
        ret_exp.local_pred = {}
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label],
             used_features,
             test_result) = self.base.testing_explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
                alpha=alpha)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp, test_result

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score, weights=None, original_data=None):
        wrapped_comparison = BlackBoxWrapper(comparison_model)
        if self.encoder is not None:
            discretize_call = self.encoder.transform
        else:
            discretize_call = lambda x: x
        if weights is not None and weights in 'euclidean':
            if comparison_data is None:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(self.perturbed_data))
            else:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(comparison_data).to_numpy())
        elif weights is not None and weights in 'nearest-neighbors':
            if comparison_data is None:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(self.perturbed_data))
            else:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(comparison_data).to_numpy())
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        if (comparison_model is None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        #elif (comparison_model is None) and (comparison_data is not None):
        #    return comparison_method(self._blackbox_classification['perturbed'],
        #                             self._surrogate_classification['perturbed'],
        #                             sample_weight=weights)

        elif (comparison_model is not None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        elif (comparison_model is not None) and (comparison_data is not None):
            return comparison_method(wrapped_comparison.predict(comparison_data),
                                     self.predict(comparison_data),
                                     sample_weight=weights)

