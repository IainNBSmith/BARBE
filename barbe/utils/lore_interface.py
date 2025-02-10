import os

import pandas as pd

import numpy as np
import scipy as sp
from sklearn.metrics import pairwise_distances
from numpy.random import RandomState

from LORE.lore import *
from barbe.perturber import BarbePerturber
from barbe.utils.evaluation_measures import *
from barbe.utils.bbmodel_interface import *
import scipy as sp
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import copy
from barbe.discretizer import CategoricalEncoder
import warnings
from LORE import pyyadt


class LoreExplainer:
    # TODO: get this finished and running LORE trials
    def __init__(self, training_data):
        self._predictor = None
        self.perturbed_data = None
        self.perturbed_lore = None
        self.encoder = CategoricalEncoder(ordinal_encoding=False)
        self.encoder.fit(training_data.drop('target', inplace=False, axis=1, errors='ignore').copy())

    def _dist_perturbations(self,
                            data_row,
                            num_samples,
                            barbe_pert_model='normal',
                            barbe_dev_scaling=1):
        # TODO: NOTE - magnify the distribution size to get results similar to them on lime2, which is not good...
        self.training_data = pd.DataFrame(self.training_data, columns=self.feature_names)
        bp = BarbePerturber(training_data=self.training_data,
                            perturbation_type=barbe_pert_model,
                            dev_scaling_factor=barbe_dev_scaling)
        data = bp.produce_perturbation(num_samples, data_row=data_row)
        inverse = data.copy()
        return data.to_numpy(), inverse.to_numpy()

    def _prepare_dataset(self, df, df_labels, input_name="data"):

        # Features Categorization
        columns = df.columns.tolist()
        possible_outcomes = list(np.unique(df_labels))

        class_name = 'target'
        #discretizer = CategoricalEncoder()
        #discretizer.fit(training_data=df.drop(class_name, axis=1))

        # TODO: continue from here using our discretizer to get details
        type_features, features_type = recognize_features_type(df, class_name)
        discrete, continuous = set_discrete_continuous(columns, type_features, class_name,
                                                       discrete=None,
                                                       continuous=None)

        columns_tmp = list(columns)
        columns_tmp.remove(class_name)
        idx_features = {i: col for i, col in enumerate(columns_tmp)}

        # Dataset Preparation for Scikit Algorithms
        df_le, label_encoder = label_encode(df, discrete)
        X = df_le.loc[:, df_le.columns != class_name].values
        y = df_le[class_name].values

        dataset = {
            'name': input_name,
            'df': df,
            'columns': list(columns),
            'class_name': class_name,
            'possible_outcomes': possible_outcomes,
            'type_features': type_features,
            'features_type': features_type,
            'discrete': discrete,
            'continuous': continuous,
            'idx_features': idx_features,
            'label_encoder': label_encoder,
            'X': X,
            'y': y,
        }

        return dataset

    def _explain(self, idx_record2explain, X2E, dataset, blackbox,
                 ng_function=genetic_neighborhood,  # generate_random_data, #genetic_neighborhood, random_neighborhood
                 dist_num_perturbations=1000,
                 dist_pert_model='normal',
                 dist_dev_scaling=1,
                 discrete_use_probabilities=False,
                 continuous_function_estimation=False,
                 returns_infos=False, path='./', sep=';', log=False):
        if ng_function is None:
            ng_function = genetic_neighborhood

        random.seed(0)
        class_name = dataset['class_name']
        columns = dataset['columns']
        discrete = dataset['discrete']
        continuous = dataset['continuous']
        features_type = dataset['features_type']
        label_encoder = dataset['label_encoder']
        possible_outcomes = dataset['possible_outcomes']

        # Dataset Preprocessing
        dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                             discrete_use_probabilities, continuous_function_estimation)

        dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

        # Generate Neighborhood
        if ng_function is not "barbe":
            dfZ, Z = ng_function(dfZ, x, blackbox, dataset)
        else:
            self.training_data = dataset
            dfZ, Z = self._dist_perturbations(x, dist_num_perturbations,
                                              barbe_pert_model=dist_pert_model,
                                              barbe_dev_scaling=dist_dev_scaling)

        # Build Decision Tree
        #print("DATA NAME: ", dataset['name'])
        #print("DATA PATH: ", os.getcwd())
        #print("Passed Path: ", path)
        dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous,
                                filename=dataset['name'], path=path, sep=sep, log=log)

        # Apply Black Box and Decision Tree on instance to explain
        print("COL:", columns)
        print("x:", x)
        input_row0 = pd.DataFrame(columns=columns, index=[0])
        input_row0 = input_row0.drop('target', axis=1)
        print("input_row0:",input_row0)
        input_row0.iloc[0] = x.to_numpy().reshape((1, -1))
        bb_outcome = blackbox.predict(input_row0)[0]

        dfx = build_df2explain(blackbox, x.to_numpy().reshape(1, -1), dataset).to_dict('records')[0]
        cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

        # Apply Black Box and Decision Tree on neighborhood
        dfZ = dfZ.drop('target', axis=1, errors='ignore')
        y_pred_bb = blackbox.predict(dfZ)
        y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                               discrete, continuous)

        def predict(X):
            y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
            return y, ln

        # Update labels if necessary
        if class_name in label_encoder:
            cc_outcome = label_encoder[class_name].transform(np.array([[cc_outcome]]))[0]

        if class_name in label_encoder:
            y_pred_cc = label_encoder[class_name].transform([[y] for y in y_pred_cc])

        # Extract Coutnerfactuals
        diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
        counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                     class_name, continuous, features_type)

        explanation = (rule, counterfactuals)

        infos = {
            'bb_outcome': bb_outcome,
            'cc_outcome': cc_outcome,
            'y_pred_bb': y_pred_bb,
            'y_pred_cc': y_pred_cc,
            'dfZ': dfZ,
            'Z': Z,
            'dt': dt,
            'tree_path': tree_path,
            'leaf_nodes': leaf_nodes,
            'diff_outcome': diff_outcome,
            'predict': predict,
        }
        #print(dt.graph)

        if returns_infos:
            return explanation, infos

        return explanation

    def explain(self, input_data, input_index, df, df_labels, blackbox, **kwargs):
        dataset = self._prepare_dataset(df, df_labels)
        exp, infos = self._explain(input_index, input_data, dataset, blackbox,
                                   returns_infos=True,
                                   **kwargs)

        self.perturbed_data = infos['Z']
        self.perturbed_lore = infos['dfZ']
        self._predictor = infos['predict']
        return exp, infos

    def predict(self, X):
        if self._predictor is not None:
            y, ln = self._predictor(X)
            assert ValueError("This is the ln: " + str(ln) + " and this is y: " + str(y))
            return y
        return [None]

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score, weights=None, original_data=None):
        wrapped_comparison = BlackBoxWrapper(comparison_model)
        discretize_call = self.encoder.transform
        if weights is not None and weights in 'euclidean':
            if comparison_data is None:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(self.perturbed_data))
            else:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(comparison_data.copy()).to_numpy())
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
        #discretize_call = lambda x: x.to_dict('records')[0]
        if (comparison_model is None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_lore),
                                     self.predict(self.perturbed_lore.to_dict('records')),
                                     sample_weight=weights)
        #elif (comparison_model is None) and (comparison_data is not None):
        #    return comparison_method(self._blackbox_classification['perturbed'],
        #                             self._surrogate_classification['perturbed'],
        #                             sample_weight=weights)

        elif (comparison_model is not None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_lore),
                                     self.predict(self.perturbed_lore.copy().to_dict('records')),
                                     sample_weight=weights)
        elif (comparison_model is not None) and (comparison_data is not None):
            #print("DATA: ", comparison_data.to_dict('records'))
            return comparison_method(wrapped_comparison.predict(comparison_data.copy()),
                                     self.predict(comparison_data.copy().to_dict('records')),
                                     sample_weight=weights)
