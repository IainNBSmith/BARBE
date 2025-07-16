# IAIN make this file get and store results for each of the fidelity values we will check
#  against each other for each distribution type + lime1's distribution
import dill
from itertools import product

# experiments to see which of Lime, Uniform, Standard-Normal, Multi-Normal, Clustered-Normal, t-Distribution, Chauchy
#  perform the best in terms of fidelity values

# pert_fidelity = barbe[i].fidelity() -> training accuracy
# train_fidelity = barbe[i].fidelity(training_data, bbmodel) -> single blind / validation accuracy
# test_fidelity = barbe[i].fidelity(bbmodel_data, bbmodel) -> double blind / testing accuracy

from barbe.utils.lime_interface import LimeNewPert, VAELimeNewPert
from barbe.utils.lore_interface import LoreExplainer
from barbe.utils.bbmodel_interface import BlackBoxWrapper
from barbe.utils.fieap_interface import FIEAPClassifier
from barbe.discretizer import CategoricalEncoder
from datetime import datetime
import os
import dice_ml
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import barbe.tests.tests_config as tests_config
import random
from numpy.random import RandomState
from barbe.explainer import BARBE
from barbe.utils.evaluation_measures import EuclideanDistanceInterval
import numpy as np
import traceback
import matplotlib.pyplot as plt
from barbe.utils.simulation_datasets import *
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lime1.lime_tabular import LimeTabularExplainer
import pickle
from barbe.utils.lore_interface import LoreExplainer
'''
Remember the selling points of BARBE:
- More customizable than LIME **
- Potentially works better than LIME with less input data and perturbations
- More plain language explanations than LIME
- Works on some cases that LIME does not
- Visual interface to interact with the models
    - Adding to this with the ability to manipulate SigDirect could be ++
    - That interface should be simple and never get too over encumbered to use

'''


# (October 16) TODO: Make the experiment also run Lime and use the Lime perturber for BARBE
# (October 16) TODO: Add experiments for more advanced data like loans and others you find
# (October 16) TODO: Add experiment comparing the feature importance for the RF to BARBE/LIME
# (October 16) TODO: Add experiments for more advanced models e.g. NN (should also be a test), KAN, etc.
# Lime requires probabilities

def _lime_assess_predictions(lime_models, lime_pert_data, iris_training, iris_test):
    temp_lime_pert_predictions = None
    temp_lime_single_predictions = None
    temp_lime_double_predictions = None
    temp_lime_pert_proba = None
    temp_lime_single_proba = None
    temp_lime_double_proba = None
    #print(lime_models)
    for label in lime_models:
        if temp_lime_pert_proba is None:
            temp_lime_pert_proba = lime_models[label].predict(lime_pert_data)
            temp_lime_single_proba = lime_models[label].predict(iris_training)
            temp_lime_double_proba = lime_models[label].predict(iris_test)
            temp_lime_pert_predictions = np.array([label for _ in range(lime_pert_data.shape[0])])
            temp_lime_single_predictions = np.array([label for _ in range(iris_training.shape[0])])
            temp_lime_double_predictions = np.array([label for _ in range(iris_test.shape[0])])
        else:
            lime_pert_replace = [(True if temp_lime_pert_proba[i] < lime_models[label].predict(lime_pert_data[i, :].reshape(1,-1))
                                  else False) for i in range(lime_pert_data.shape[0])]
            lime_single_replace = [(True if temp_lime_pert_proba[i] < lime_models[label].predict(iris_training.iloc[i].to_numpy().reshape(1,-1))
                                    else False) for i in range(iris_training.shape[0])]
            lime_double_replace = [(True if temp_lime_pert_proba[i] < lime_models[label].predict(iris_test.iloc[i].to_numpy().reshape(1,-1))
                                    else False) for i in range(iris_test.shape[0])]
            temp_lime_pert_proba[lime_pert_replace] = lime_models[label].predict(lime_pert_data)[
                lime_pert_replace]
            temp_lime_single_proba[lime_single_replace] = lime_models[label].predict(iris_training)[
                lime_single_replace]
            temp_lime_double_proba[lime_double_replace] = lime_models[label].predict(iris_test)[
                lime_double_replace]
            # IAIN lime1 tends to produce data all with the same label
            temp_lime_pert_predictions[lime_pert_replace] = label
            temp_lime_single_predictions[lime_single_replace] = label
            temp_lime_double_predictions[lime_double_replace] = label
    return temp_lime_pert_predictions, temp_lime_single_predictions, temp_lime_double_predictions


def euclidean_distance(training_data, index):
    dist = EuclideanDistanceInterval()
    np.set_printoptions(precision=18)
    print(dist.get_euclidean_distance(training_data, training_data.iloc[index]))
    print(dist.get_nearest_neighbor_distance(training_data, training_data.iloc[index]))


def lore_distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, dev_scaling=1.0,
                                 local_test_end=20, data_name="data", n_perturbations=100,
                                 use_barbe_perturbations=False):
    # TODO: make this set up folds to run the experiments over and then be more generic

    # LIME requires more steps preprocess the data here
    #cat_encoder = CategoricalEncoder(ordinal_encoding=False)
    #iris_training = cat_encoder.fit_transform(training_data=iris_training)
    #iris_perturb = cat_encoder.transform(iris_perturb)
    #iris_test = cat_encoder.transform(iris_test)
    #print(cat_encoder._encoder_key)

    # TODO: discuss the results from a overfit black box model
    if pre_trained_model is None:
        #bbmodel = RandomForestClassifier(n_estimators=5,
        #                                 max_depth=4,
        #                                 min_samples_split=10,
        #                                 min_samples_leaf=3,
        #                                 bootstrap=True,
        #                                 random_state=301257)
        bbmodel = MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257)
        bbmodel.fit(iris_training, iris_training_label)
        #print(confusion_matrix(iris_training_label, bbmodel.predict(iris_training)))
        #assert False
        #print(confusion_matrix(iris_training_label, bbmodel.predict(iris_training)))
        #print(bbmodel.predict_proba(iris_training))
        #assert False
    else:
        bbmodel = pre_trained_model

    #print(np.eye(3, 3)[np.argmax(bbmodel.predict_proba(iris_training), axis=1)])
    #assert False
    random.seed(516231)
    part_training = int(iris_training.shape[0] // 3)  # specifically for the LOAN
    iris_wb_training = iris_training.iloc[0:part_training]

    barbe_dist = ['standard-normal', 'normal', 'uniform', 'cauchy', 't-distribution'] if use_barbe_perturbations else ['pert-lime']
    # barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end)]
    fidelity_single_blind = [[] for _ in range(local_test_end)]
    fidelity_double_blind = [[] for _ in range(local_test_end)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end)]

    fidelity_single_diff_n = [[] for _ in range(local_test_end)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end)]

    hit_rate = [[] for _ in range(local_test_end)]

    iris_category_features = list(np.where(np.isin(np.array(list(iris_training)), discrete_features))[0])
    print("Processed Categories: ", iris_category_features)

    #iris_perturb['target'] = bbmodel.predict(iris_perturb)
    iris_test['target'] = bbmodel.predict(iris_test.drop('target', axis=1, errors='ignore'))
    iris_wb_training['target'] = bbmodel.predict(iris_wb_training.drop('target', axis=1, errors='ignore'))
    iris_training['target'] = bbmodel.predict(iris_training.drop('target', axis=1, errors='ignore'))
    for c in np.unique(iris_training['target']):
        class_position = np.where(iris_training['target'] == c)[0]
        print("CLASS POSITION: ", class_position)
        print("CLASS VALUE: ", c)
        iris_wb_training = iris_wb_training.append(iris_training.iloc[class_position[0:2]], ignore_index=True)
    iris_training.drop('target', inplace=True, axis=1)
    iris_test.drop('target', inplace=True, axis=1, errors='ignore')
    for i in range(local_test_end):  # use when removing LIME
        #pert_row = iris_perturb.drop('target', inplace=False, axis=1).iloc[i]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        iris_wb_test = iris_wb_test.drop('target', inplace=False, axis=1, errors='ignore')
        for distribution in barbe_dist:

            try:
                iris_numpy = iris_training.to_numpy()
                #print(np.unique(bbmodel.predict(iris_training)))
                #print(np.unique(iris_training_label))
                #print(np.unique(bbmodel.predict(iris_test)))
                #assert False
                explainer = LoreExplainer(iris_wb_training)
                #explainer = LoreExplainer(training_data=iris_wb_training,
                #                          training_labels=iris_training_label,
                #                          feature_names=list(iris_training),
                #                          discretizer=lime_discretizer,
                #                          discretize_continuous=False,
                #                          sample_around_instance=True)
                                         #categorical_features=iris_category_values,
                                         #categorical_names=iris_category_features)
                print("TRAINING LABELS: ", list(iris_wb_training['target']))
                _ = explainer.explain(input_data=iris_test,
                                      input_index=i,
                                      df=iris_wb_training,
                                      df_labels=list(bbmodel.predict(iris_training)),
                                      blackbox=bbmodel,
                                      discrete_use_probabilities=True)  # IAIN see if this works

                pert_row = iris_test.iloc[i]
                input_row = pd.DataFrame(columns=list(iris_training), index=[0])
                input_row.iloc[0] = pert_row.to_numpy().reshape((1, -1))
                # .to_numpy().reshape(1, -1)
                if (str(explainer.predict([input_row.to_dict('records')[0]])[0]) !=
                        str(bbmodel.predict(input_row)[0])):
                    print(explainer.predict([input_row.to_dict('records')[0]]), bbmodel.predict(input_row))
                    assert False

                temp_pert = explainer.get_surrogate_fidelity(comparison_model=bbmodel)
                temp_single_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training.drop('target', axis=1, errors='ignore'),
                                                               weights=None,
                                                               original_data=pert_row)
                temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test.drop('target', axis=1, errors='ignore'),
                                                               weights=None,
                                                               original_data=pert_row)
                fidelity_pert[i].append(temp_pert)
                fidelity_single_blind[i].append(temp_single_f)
                fidelity_double_blind[i].append(temp_double_f)
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training.drop('target', axis=1, errors='ignore'),
                                                               weights='euclidean',
                                                               original_data=pert_row)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test.drop('target', axis=1, errors='ignore'),
                                                               weights='euclidean',
                                                               original_data=pert_row)
                fidelity_single_blind_e[i].append(temp_single)
                fidelity_double_blind_e[i].append(temp_double)

                fidelity_single_diff_e[i].append(temp_single - temp_single_f)
                fidelity_double_diff_e[i].append(temp_double - temp_double_f)
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training,
                                                               weights='nearest',
                                                               original_data=pert_row)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test,
                                                               weights='nearest',
                                                               original_data=pert_row)
                fidelity_single_blind_n[i].append(temp_single)
                fidelity_double_blind_n[i].append(temp_double)

                fidelity_single_diff_n[i].append(temp_single - temp_single_f)
                fidelity_double_diff_n[i].append(temp_double - temp_double_f)
                hit_rate[i].append(1)
            except:
                if False:
                    for ar_v in [fidelity_single_blind,
                                 fidelity_double_blind,
                                 fidelity_single_blind_e,
                                 fidelity_double_blind_e,
                                 fidelity_single_blind_n,
                                 fidelity_single_blind_n,
                                 fidelity_double_diff_e,
                                 fidelity_single_diff_e,
                                 fidelity_double_diff_n,
                                 fidelity_single_diff_n]:
                        if len(ar_v[i]) < len(fidelity_pert[i]):
                            print(traceback.format_exc())
                            assert False
                print(traceback.format_exc())
                if (str(explainer.predict([input_row.to_dict('records')[0]])[0]) ==
                        str(bbmodel.predict(input_row)[0])):
                    print(traceback.format_exc())
                    assert False
                temp_pert = None
                temp_single = None
                temp_double = None
                fidelity_pert[i].append(temp_pert)
                fidelity_single_blind[i].append(temp_single)
                fidelity_double_blind[i].append(temp_double)
                fidelity_single_blind_e[i].append(temp_single)
                fidelity_double_blind_e[i].append(temp_double)
                fidelity_single_blind_n[i].append(temp_single)
                fidelity_double_blind_n[i].append(temp_double)
                fidelity_double_diff_e[i].append(None)
                fidelity_single_diff_e[i].append(None)
                fidelity_double_diff_n[i].append(None)
                fidelity_single_diff_n[i].append(None)

                hit_rate[i].append(0)

    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate"]]

    for fidelity_pert, fidelity_single_blind, fidelity_double_blind, run in \
            [(fidelity_pert, fidelity_single_blind, fidelity_double_blind, 'regular'),
             (fidelity_pert, fidelity_single_blind_e, fidelity_double_blind_e, 'euclidean'),
             (fidelity_pert, fidelity_single_blind_n, fidelity_double_blind_n, 'nearest neighbors'),
             (fidelity_pert, fidelity_single_diff_e, fidelity_double_diff_e, 'euc. diff.'),
             (fidelity_pert, fidelity_single_diff_n, fidelity_double_diff_n, 'nn. diff.')]:
        for j in range(len(barbe_dist)):
            temp_pert_acc = 0
            temp_single_acc = 0
            temp_double_acc = 0
            mean_count = 0
            for i in range(local_test_end):
                if fidelity_pert[i][j] is not None:
                    mean_count += 1
                    temp_pert_acc += fidelity_pert[i][j]
                    temp_single_acc += fidelity_single_blind[i][j]
                    temp_double_acc += fidelity_double_blind[i][j]
            temp_pert_std = 0
            temp_single_std = 0
            temp_double_std = 0
            for i in range(local_test_end):
                if fidelity_pert[i][j] is not None:
                    temp_pert_std += (fidelity_pert[i][j] - temp_pert_acc / mean_count) ** 2
                    temp_single_std += (fidelity_single_blind[i][j] - temp_single_acc / mean_count) ** 2
                    temp_double_std += (fidelity_double_blind[i][j] - temp_double_acc / mean_count) ** 2
            temp_pert_std = (temp_pert_std / (mean_count - 1))
            temp_single_std = (temp_single_std / (mean_count - 1))
            temp_double_std = (temp_double_std / (mean_count - 1))
            # print(barbe_dist[j])
            if mean_count != 0:
                # add standard deviations as info
                if (len(averages_print) - 1) / 3 <= j:
                    averages_print.append([barbe_dist[j], "Perturbed"])
                    averages_print.append([barbe_dist[j], "Single Blind"])
                    averages_print.append([barbe_dist[j], "Double Blind"])
                averages_print[(j * 3) + 1].append(temp_pert_acc / mean_count)
                averages_print[(j * 3) + 1].append(temp_pert_std)
                averages_print[(j * 3) + 2].append(temp_single_acc / mean_count)
                averages_print[(j * 3) + 2].append(temp_single_std)
                averages_print[(j * 3) + 3].append(temp_double_acc / mean_count)
                averages_print[(j * 3) + 3].append(temp_double_std)
    for j in range(len(barbe_dist)):
        average_hits = 0
        for i in range(local_test_end):
            average_hits += hit_rate[i][j]
        average_hits /= local_test_end
        averages_print[(j * 3) + 1].append(average_hits)
        averages_print[(j * 3) + 2].append(0)
        averages_print[(j * 3) + 3].append(0)


    pd.DataFrame(averages_print).to_csv("Results/lore_" + "_".join([data_name,
                                                               "nruns" + str(local_test_end),
                                                               "nperturb" + str(n_perturbations),
                                                               "barbe" + str(use_barbe_perturbations),
                                                               "devscalin" + str(dev_scaling)]) + "_results.csv")



def lime_distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, dev_scaling=1.0,
                                 local_test_end=20, data_name="data", n_perturbations=100,
                                 lime_version=LimeNewPert, lime_discretizer='decile',
                                 use_barbe_perturbations=False):
    # TODO: make this set up folds to run the experiments over and then be more generic

    # LIME requires more steps preprocess the data here
    #cat_encoder = CategoricalEncoder(ordinal_encoding=False)
    #iris_training = cat_encoder.fit_transform(training_data=iris_training)
    #iris_perturb = cat_encoder.transform(iris_perturb)
    #iris_test = cat_encoder.transform(iris_test)
    #print(cat_encoder._encoder_key)

    # TODO: discuss the results from a overfit black box model
    if pre_trained_model is None:
        bbmodel = RandomForestClassifier(n_estimators=5,
                                         max_depth=4,
                                         min_samples_split=10,
                                         min_samples_leaf=3,
                                         bootstrap=True,
                                         random_state=301257)
        bbmodel = MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257)
        bbmodel.fit(iris_training, iris_training_label)
        #print(confusion_matrix(iris_training_label, bbmodel.predict(iris_training)))
        #print(bbmodel.predict_proba(iris_training))
        #assert False
    else:
        bbmodel = pre_trained_model

    #print(np.eye(3, 3)[np.argmax(bbmodel.predict_proba(iris_training), axis=1)])
    #assert False
    random.seed(516231)
    part_training = int(iris_training.shape[0] // 3)  # specifically for the LOAN
    iris_wb_training = iris_training.iloc[0:part_training]

    barbe_dist = ['standard-normal', 'normal', 'uniform', 'cauchy', 't-distribution'] if use_barbe_perturbations else ['pert-lime']
    # barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end)]
    fidelity_single_blind = [[] for _ in range(local_test_end)]
    fidelity_double_blind = [[] for _ in range(local_test_end)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end)]

    fidelity_single_diff_n = [[] for _ in range(local_test_end)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end)]

    hit_rate = [[] for _ in range(local_test_end)]

    iris_category_features = list(np.where(np.isin(np.array(list(iris_training)), discrete_features))[0])
    print("Processed Categories: ", iris_category_features)

    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:

            try:
                iris_numpy = iris_training.to_numpy()
                #print(np.unique(bbmodel.predict(iris_training)))
                #print(np.unique(iris_training_label))
                #print(np.unique(bbmodel.predict(iris_test)))
                #assert False
                explainer = lime_version(training_data=iris_wb_training,
                                         training_labels=iris_training_label,
                                         feature_names=list(iris_training),
                                         discretizer=lime_discretizer,
                                         discretize_continuous=False,
                                         sample_around_instance=True)
                                         #categorical_features=iris_category_values,
                                         #categorical_names=iris_category_features)
                _ = explainer.explain_instance(data_row=pert_row,
                                               predict_fn=lambda x: (np.eye(15, 15)[np.argmax(bbmodel.predict_proba(x), axis=1)]),
                                               labels= (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 14, ),
                                               num_features=iris_wb_training.shape[1],
                                               num_samples=n_perturbations,
                                               barbe_mode=use_barbe_perturbations,
                                               barbe_pert_model=distribution,
                                               barbe_dev_scaling=dev_scaling)  # IAIN see if this works

                input_row = pd.DataFrame(columns=list(iris_training), index=[0])
                input_row.iloc[0] = pert_row.to_numpy().reshape((1, -1))
                # .to_numpy().reshape(1, -1)
                if (str(explainer.predict(pert_row)[0]) !=
                        str(bbmodel.predict(input_row)[0])):
                    print(explainer.predict(pert_row), bbmodel.predict(input_row))
                    assert False

                temp_pert = explainer.get_surrogate_fidelity(comparison_model=bbmodel)
                temp_single_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training,
                                                               weights=None,
                                                               original_data=pert_row)
                temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test,
                                                               weights=None,
                                                               original_data=pert_row)
                fidelity_pert[i].append(temp_pert)
                fidelity_single_blind[i].append(temp_single_f)
                fidelity_double_blind[i].append(temp_double_f)
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training,
                                                               weights='euclidean',
                                                               original_data=pert_row)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test,
                                                               weights='euclidean',
                                                               original_data=pert_row)
                fidelity_single_blind_e[i].append(temp_single)
                fidelity_double_blind_e[i].append(temp_double)

                fidelity_single_diff_e[i].append(temp_single - temp_single_f)
                fidelity_double_diff_e[i].append(temp_double - temp_double_f)
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training,
                                                               weights='nearest',
                                                               original_data=pert_row)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test,
                                                               weights='nearest',
                                                               original_data=pert_row)
                fidelity_single_blind_n[i].append(temp_single)
                fidelity_double_blind_n[i].append(temp_double)

                fidelity_single_diff_n[i].append(temp_single - temp_single_f)
                fidelity_double_diff_n[i].append(temp_double - temp_double_f)
                hit_rate[i].append(1)
            except:
                if False:
                    for ar_v in [fidelity_single_blind,
                                 fidelity_double_blind,
                                 fidelity_single_blind_e,
                                 fidelity_double_blind_e,
                                 fidelity_single_blind_n,
                                 fidelity_single_blind_n,
                                 fidelity_double_diff_e,
                                 fidelity_single_diff_e,
                                 fidelity_double_diff_n,
                                 fidelity_single_diff_n]:
                        if len(ar_v[i]) < len(fidelity_pert[i]):
                            print(traceback.format_exc())
                            assert False
                #print(traceback.format_exc())
                #assert False
                temp_pert = None
                temp_single = None
                temp_double = None
                fidelity_pert[i].append(temp_pert)
                fidelity_single_blind[i].append(temp_single)
                fidelity_double_blind[i].append(temp_double)
                fidelity_single_blind_e[i].append(temp_single)
                fidelity_double_blind_e[i].append(temp_double)
                fidelity_single_blind_n[i].append(temp_single)
                fidelity_double_blind_n[i].append(temp_double)
                fidelity_double_diff_e[i].append(None)
                fidelity_single_diff_e[i].append(None)
                fidelity_double_diff_n[i].append(None)
                fidelity_single_diff_n[i].append(None)

                hit_rate[i].append(0)

    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate"]]

    for fidelity_pert, fidelity_single_blind, fidelity_double_blind, run in \
            [(fidelity_pert, fidelity_single_blind, fidelity_double_blind, 'regular'),
             (fidelity_pert, fidelity_single_blind_e, fidelity_double_blind_e, 'euclidean'),
             (fidelity_pert, fidelity_single_blind_n, fidelity_double_blind_n, 'nearest neighbors'),
             (fidelity_pert, fidelity_single_diff_e, fidelity_double_diff_e, 'euc. diff.'),
             (fidelity_pert, fidelity_single_diff_n, fidelity_double_diff_n, 'nn. diff.')]:
        for j in range(len(barbe_dist)):
            temp_pert_acc = 0
            temp_single_acc = 0
            temp_double_acc = 0
            mean_count = 0
            for i in range(local_test_end):
                if fidelity_pert[i][j] is not None:
                    mean_count += 1
                    temp_pert_acc += fidelity_pert[i][j]
                    temp_single_acc += fidelity_single_blind[i][j]
                    temp_double_acc += fidelity_double_blind[i][j]
            temp_pert_std = 0
            temp_single_std = 0
            temp_double_std = 0
            for i in range(local_test_end):
                if fidelity_pert[i][j] is not None:
                    temp_pert_std += (fidelity_pert[i][j] - temp_pert_acc / mean_count) ** 2
                    temp_single_std += (fidelity_single_blind[i][j] - temp_single_acc / mean_count) ** 2
                    temp_double_std += (fidelity_double_blind[i][j] - temp_double_acc / mean_count) ** 2
            temp_pert_std = (temp_pert_std / (mean_count - 1))
            temp_single_std = (temp_single_std / (mean_count - 1))
            temp_double_std = (temp_double_std / (mean_count - 1))
            # print(barbe_dist[j])
            if mean_count != 0:
                # add standard deviations as info
                if (len(averages_print) - 1) / 3 <= j:
                    averages_print.append([barbe_dist[j], "Perturbed"])
                    averages_print.append([barbe_dist[j], "Single Blind"])
                    averages_print.append([barbe_dist[j], "Double Blind"])
                averages_print[(j * 3) + 1].append(temp_pert_acc / mean_count)
                averages_print[(j * 3) + 1].append(temp_pert_std)
                averages_print[(j * 3) + 2].append(temp_single_acc / mean_count)
                averages_print[(j * 3) + 2].append(temp_single_std)
                averages_print[(j * 3) + 3].append(temp_double_acc / mean_count)
                averages_print[(j * 3) + 3].append(temp_double_std)
    for j in range(len(barbe_dist)):
        average_hits = 0
        for i in range(local_test_end):
            average_hits += hit_rate[i][j]
        average_hits /= local_test_end
        averages_print[(j * 3) + 1].append(average_hits)
        averages_print[(j * 3) + 2].append(0)
        averages_print[(j * 3) + 3].append(0)


    pd.DataFrame(averages_print).to_csv("Results/lime_" + "_".join([data_name,
                                                               "nruns" + str(local_test_end),
                                                               "nperturb" + str(n_perturbations),
                                                               "barbe" + str(use_barbe_perturbations),
                                                               "devscalin" + str(dev_scaling)]) + "_results.csv")


def counterfactual_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                              discrete_features=None, restricted_features=None, use_negation_rules=True,
                              local_test_end=20, data_name="data", n_perturbations=100, n_bins=5, dev_scaling=10):
    # TODO: make this set up folds to run the experiments over and then be more generic
    if restricted_features is None:
        restricted_features = ['Gender', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

    if pre_trained_model is None:
        bbmodel = RandomForestClassifier(n_estimators=5,
                                         max_depth=4,
                                         min_samples_split=10,
                                         min_samples_leaf=3,
                                         bootstrap=True,
                                         random_state=301257)
        bbmodel = MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257)
        bbmodel.fit(iris_training, iris_training_label)
    else:
        bbmodel = pre_trained_model

    random.seed(516231)
    part_training = int(iris_training.shape[0] // 3)  # specifically for the LOAN
    iris_wb_training = iris_training.iloc[0:part_training]

    barbe_dist = ['standard-normal', 'normal']
    #barbe_dist = ['normal']


    from barbe.utils.evaluation_measures import FlexibleDifference

    results = pd.DataFrame(columns=['distribution', 'counter-method', 'original-class', 'explain-time', 'fidelity',
                                    'hit', 'counter-time',
                                    'c-hit-1', 'c-hit-2', 'c-hit-3', 'c-hit-4', 'c-hit-5', 'c-hit',
                                    'diff-c-1', 'diff-c-2', 'diff-c-3', 'diff-c-4', 'diff-c-5',
                                    'dens-c-1', 'dens-c-2', 'dens-c-3', 'dens-c-4', 'dens-c-5',
                                    'n-c-hit',
                                    'counter-time-r',
                                    'c-hit-1r', 'c-hit-2r', 'c-hit-3r', 'c-hit-4r', 'c-hit-5r', 'c-hit-r',
                                    'diff-c-1r', 'diff-c-2r', 'diff-c-3r', 'diff-c-4r', 'diff-c-5r',
                                    'dens-c-1r', 'dens-c-2r', 'dens-c-3r', 'dens-c-4r', 'dens-c-5r',
                                    'n-c-hit-r'
                                    ])

    diff_calc = FlexibleDifference(iris_training)

    unique_options = np.unique(bbmodel.predict(iris_training))
    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i:(i+1)]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:
            print(distribution, ": ", i)
            result_row = list()
            result_row.append(distribution)
            #print(unique_options)
            #assert False

            explainer = BARBE(training_data=iris_wb_training,
                              input_bounds=None,#[(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                              perturbation_type=distribution,
                              n_perturbations=n_perturbations,
                              dev_scaling_factor=dev_scaling,
                              learn_negation_rules=use_negation_rules,
                              n_bins=n_bins,
                              verbose=False,
                              input_sets_class=False)


            # TODO: rules for multivariate data need unique considerations
            start = time.time()
            explanation = explainer.explain(pert_row.copy(), bbmodel, ignore_errors=True)
            end = time.time()

            input_row = pd.DataFrame(columns=list(iris_training), index=[0])
            input_row.loc[0, :] = pert_row.to_numpy().reshape((1, -1))
            initial_row = input_row.copy()
            result_row.append(bbmodel.predict(input_row)[0])
            result_row.append(end-start)

            if explanation is not None:
                # TODO: run the alternative for the importance rules right after this... rather than as another thing...

                temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                 comparison_data=iris_wb_test,
                                                                 weights='euclidean',
                                                                 original_data=pert_row)
                result_row.append(temp_double_f)  # fidelity

                result_row.append(1)  # hit

                wanted_class = unique_options[0] if bbmodel.predict(input_row)[0] == unique_options[1] else unique_options[1]

                old_results = result_row.copy()
                for use_importance in [True, False]:
                    if use_importance:
                        current_sort = 'importance-rules'
                    else:
                        current_sort = 'high-rules'

                    result_row = old_results.copy()
                    result_row.insert(1, current_sort)

                    restricted_options = [None] if restricted_features is None else [None, restricted_features]
                    for restriction in restricted_options:
                        start_c = time.time()
                        counterfactual = explainer.get_counterfactuals(pert_row.copy(), bbmodel, wanted_class, 5,
                                                                       importance_counterfactuals=False,
                                                                       restricted_features=restriction,
                                                                       prioritize_importance=use_importance)
                        end_c = time.time()

                        result_row.append(end_c - start_c)

                        any_success = False
                        success_count = 0
                        exp_change_count = 0
                        total_nums = 0

                        # TODO: make each of these their own count for hits (store the full table not just the averaged results)
                        density_list = list()
                        distance_list = list()
                        for option in counterfactual[0]:
                            #print(np.array(option[0]))
                            #print(np.array(input_row.iloc[0]))
                            #print(np.array(option[0]) != np.array(input_row.iloc[0]))
                            if len(option) > 0 and np.any(np.array(option[0]) != np.array(initial_row.iloc[0])):
                                total_nums += 1
                                input_row.loc[0, :] = option
                                #print(f"OPTION {total_nums}: {explainer.predict(input_row)[0]} vs {wanted_class}")
                                if explainer.predict(input_row)[0] == wanted_class:
                                    exp_change_count += 1
                                if bbmodel.predict(input_row)[0] == wanted_class:
                                    success_count += 1
                                    result_row.append(1)
                                    any_success = True
                                else:
                                    result_row.append(0)
                                density_list.append(diff_calc.get_density_distance(initial_row, input_row))
                                distance_list.append(diff_calc.get_scaled_distance(initial_row, input_row))

                            else:
                                result_row.append(-1)
                                density_list.append(-1)
                                distance_list.append(-1)

                        if len(counterfactual[0]) < 5:
                            for _ in range(5 - len(counterfactual[0])):
                                result_row.append(-1)
                                density_list.append(-1)
                                distance_list.append(-1)


                        result_row.append(int(any_success))
                        result_row += distance_list
                        result_row += density_list
                        result_row.append(success_count)

                    #print(result_row)
                    #print(results)
                    results.loc[results.shape[0]] = result_row.copy()  # add to all records

                # delete to save space for next run
                del explainer
                del explanation

            else:
                result_row.insert(1, 'importance-rules')
                result_row += [-1, 0] + [-1 for _ in range(36)]
                results.loc[results.shape[0]] = result_row
                result_row[0] = 'high-rules'
                results.loc[results.shape[0]] = result_row

    results.to_csv("Results/" + "_".join([data_name,
                                          "nruns" + str(local_test_end),
                                          "nbins" + str(n_bins),
                                          "nperturb" + str(n_perturbations),
                                          "devscalin" + str(dev_scaling)]) + "_counterfactuals_results.csv")


def lore_counterfactual_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, local_test_end=20, data_name="data", n_perturbations=100,
                                 n_bins=5, dev_scaling=10, restricted_features=None):
    # TODO: make this set up folds to run the experiments over and then be more generic

    if pre_trained_model is None:
        bbmodel = RandomForestClassifier(n_estimators=5,
                                         max_depth=4,
                                         min_samples_split=10,
                                         min_samples_leaf=3,
                                         bootstrap=True,
                                         random_state=301257)
        bbmodel = MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257)
        bbmodel.fit(iris_training, iris_training_label)
    else:
        bbmodel = pre_trained_model

    random.seed(516231)
    part_training = int(iris_training.shape[0] // 3)  # specifically for the LOAN
    iris_wb_training = iris_training.iloc[0:part_training]

    barbe_dist = ['evolutionary-algorithm']


    from barbe.utils.evaluation_measures import FlexibleDifference

    results = pd.DataFrame(columns=['distribution', 'counter-method', 'original-class', 'explain-time', 'fidelity',
                                    'hit', 'counter-time',
                                    'c-hit-1', 'c-hit-2', 'c-hit-3', 'c-hit-4', 'c-hit-5', 'c-hit',
                                    'diff-c-1', 'diff-c-2', 'diff-c-3', 'diff-c-4', 'diff-c-5',
                                    'dens-c-1', 'dens-c-2', 'dens-c-3', 'dens-c-4', 'dens-c-5',
                                    'n-c-hit',
                                    'counter-time-r',
                                    'c-hit-1r', 'c-hit-2r', 'c-hit-3r', 'c-hit-4r', 'c-hit-5r', 'c-hit-r',
                                    'diff-c-1r', 'diff-c-2r', 'diff-c-3r', 'diff-c-4r', 'diff-c-5r',
                                    'dens-c-1r', 'dens-c-2r', 'dens-c-3r', 'dens-c-4r', 'dens-c-5r',
                                    'n-c-hit-r'
                                    ])

    diff_calc = FlexibleDifference(iris_training)
    iris_wb_training['target'] = bbmodel.predict(iris_wb_training)

    unique_options = np.unique(bbmodel.predict(iris_training))
    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i:(i+1)]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:
            result_row = list()
            result_row.append(distribution)
            result_row.append('lore')
            #print(unique_options)
            #assert False

            explainer = LoreExplainer(iris_wb_training)

            # TODO: rules for multivariate data need unique considerations
            start = time.time()
            explanation = explainer.explain(input_data=iris_perturb,
                                  input_index=i,
                                  df=iris_wb_training,
                                  df_labels=list(bbmodel.predict(iris_training)),
                                  blackbox=bbmodel,
                                  discrete_use_probabilities=True)  # IAIN see if this works
            end = time.time()

            input_row = pd.DataFrame(columns=list(iris_test), index=[0])
            input_row.loc[0, :] = pert_row.to_numpy().reshape((1, -1))
            initial_row = input_row.copy()
            result_row.append(bbmodel.predict(input_row)[0])
            result_row.append(end-start)

            #print(explainer.predict([input_row.to_dict('records')[0]])[i], bbmodel.predict(input_row)[0])
            #assert False
            if explainer.predict([input_row.to_dict('records')[0]])[0] == bbmodel.predict(input_row)[0]:
                try:
                    temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_wb_test,
                                                                     weights='euclidean',
                                                                     original_data=pert_row)
                except:
                    temp_double_f = -1
                result_row.append(temp_double_f)  # fidelity

                result_row.append(1)  # hit

                wanted_class = unique_options[0] if bbmodel.predict(input_row)[0] == unique_options[1] else unique_options[1]

                restricted_options = [[]] if restricted_features is None else [[], restricted_features]
                for restriction in restricted_options:
                    start_c = time.time()
                    counterfactual = explainer.get_counterfactual(pert_row.copy(), restricted_features=restriction)
                    end_c = time.time()

                    result_row.append(end_c - start_c)

                    any_success = False
                    success_count = 0
                    exp_change_count = 0
                    total_nums = 0

                    # TODO: make each of these their own count for hits (store the full table not just the averaged results)
                    density_list = list()
                    distance_list = list()
                    for option in counterfactual:
                        #print(np.array(option[0]))
                        #print(np.array(input_row.iloc[0]))
                        #print(np.array(option[0]) != np.array(input_row.iloc[0]))
                        if len(option) > 0 and np.any(np.array(option) != np.array(initial_row.iloc[0])):
                            total_nums += 1
                            #input_row = option
                            #print(f"OPTION {total_nums}: {explainer.predict(input_row)[0]} vs {wanted_class}")
                            #if explainer.predict(input_row.to_numpy().reshape((1,-1)))[0] == wanted_class:
                            #    exp_change_count += 1
                            if bbmodel.predict(option)[0] == wanted_class:
                                success_count += 1
                                result_row.append(1)
                                any_success = True
                            else:
                                result_row.append(0)
                            density_list.append(diff_calc.get_density_distance(initial_row, option))
                            distance_list.append(diff_calc.get_scaled_distance(initial_row, option))

                        else:
                            result_row.append(-1)
                            density_list.append(-1)
                            distance_list.append(-1)

                    if len(counterfactual) < 5:
                        for _ in range(5-len(counterfactual)):
                            result_row.append(-1)
                            density_list.append(-1)
                            distance_list.append(-1)

                    if len(counterfactual) > 5:
                        result_row = result_row[:(5-len(counterfactual))]
                        density_list = density_list[:5]
                        distance_list = distance_list[:5]

                    result_row.append(int(any_success))
                    result_row += distance_list
                    result_row += density_list
                    result_row.append(success_count)

                results.loc[results.shape[0]] = result_row  # add to all records

                # delete to save space for next run
                del explainer
                del explanation

            else:
                result_row += [-1, 0] + [-1 for _ in range(36)]
                results.loc[results.shape[0]] = result_row

    results.to_csv("Results/" + "_".join([data_name,
                                          "nruns" + str(local_test_end),
                                          "nbins" + str(n_bins),
                                          "nperturb" + str(n_perturbations),
                                          "devscalin" + str(dev_scaling)]) + "_counterfactuals_results.csv")

def dice_counterfactual_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, local_test_end=20, data_name="data", n_perturbations=100,
                                 n_bins=5, dev_scaling=10, restricted_features=None):
    # TODO: make this set up folds to run the experiments over and then be more generic

    if pre_trained_model is None:
        bbmodel = RandomForestClassifier(n_estimators=5,
                                         max_depth=4,
                                         min_samples_split=10,
                                         min_samples_leaf=3,
                                         bootstrap=True,
                                         random_state=301257)
        bbmodel = MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257)
        bbmodel.fit(iris_training, iris_training_label)
    else:
        bbmodel = pre_trained_model

    random.seed(516231)
    part_training = int(iris_training.shape[0] // 3)  # specifically for the LOAN
    iris_wb_training = iris_training.iloc[0:part_training]

    barbe_dist = ['random', 'genetic']


    from barbe.utils.evaluation_measures import FlexibleDifference

    results = pd.DataFrame(columns=['distribution', 'counter-method', 'original-class', 'explain-time', 'fidelity',
                                    'hit', 'counter-time',
                                    'c-hit-1', 'c-hit-2', 'c-hit-3', 'c-hit-4', 'c-hit-5', 'c-hit',
                                    'diff-c-1', 'diff-c-2', 'diff-c-3', 'diff-c-4', 'diff-c-5',
                                    'dens-c-1', 'dens-c-2', 'dens-c-3', 'dens-c-4', 'dens-c-5',
                                    'n-c-hit',
                                    'counter-time-r',
                                    'c-hit-1r', 'c-hit-2r', 'c-hit-3r', 'c-hit-4r', 'c-hit-5r', 'c-hit-r',
                                    'diff-c-1r', 'diff-c-2r', 'diff-c-3r', 'diff-c-4r', 'diff-c-5r',
                                    'dens-c-1r', 'dens-c-2r', 'dens-c-3r', 'dens-c-4r', 'dens-c-5r',
                                    'n-c-hit-r'
                                    ])

    diff_calc = FlexibleDifference(iris_training)
    iris_wb_training['target'] = bbmodel.predict(iris_wb_training)


    unique_options = np.unique(bbmodel.predict(iris_training))
    bbmodel = BlackBoxWrapper(bbmodel=bbmodel, class_labels=unique_options)

    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i:(i+1)]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:
            result_row = list()
            result_row.append(distribution)
            result_row.append('lore')
            #print(unique_options)
            #assert False

            #explainer = LoreExplainer(iris_wb_training)

            start = time.time()
            #print(set(iris_training.columns).difference(set(discrete_features)))
            #print(iris_wb_training.columns)
            d = dice_ml.Data(dataframe=iris_wb_training,
                             continuous_features=list(set(iris_training.columns).difference(set(discrete_features))),
                             outcome_name='target')
            m = dice_ml.Model(model=bbmodel, backend='sklearn')
            explanation = dice_ml.Dice(d, m, method=distribution)
            #explanation = explainer.explain(input_data=iris_perturb,
            #                      input_index=i,
            #                      df=iris_wb_training,
            #                      df_labels=list(bbmodel.predict(iris_training)),
            #                      blackbox=bbmodel,
            #                      discrete_use_probabilities=True)  # IAIN see if this works
            end = time.time()

            input_row = pd.DataFrame(columns=list(iris_test), index=[0])
            input_row.loc[0, :] = pert_row.to_numpy().reshape((1, -1))
            initial_row = input_row.copy()
            result_row.append(bbmodel.predict(input_row)[0])
            result_row.append(end-start)

            #print(explainer.predict([input_row.to_dict('records')[0]])[i], bbmodel.predict(input_row)[0])
            #assert False
            if True:
                try:
                    temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_wb_test,
                                                                     weights='euclidean',
                                                                     original_data=pert_row)
                except:
                    temp_double_f = -1
                result_row.append(temp_double_f)  # fidelity

                result_row.append(1)  # hit

                wanted_class = unique_options[0] if bbmodel.predict(input_row)[0] == unique_options[1] else unique_options[1]

                restricted_options = [[]] if restricted_features is None else [[], restricted_features]
                for restriction in restricted_options:
                    start_c = time.time()
                    counterfactual = explanation.generate_counterfactuals(pert_row.copy(),
                                                                          total_CFs=5,
                                                                          features_to_vary=list(set(iris_training.columns).difference(set(restriction))))
                    end_c = time.time()

                    result_row.append(end_c - start_c)

                    counterfactuals = counterfactual.cf_examples_list[0].final_cfs_df
                    cf_list = list()
                    for i in range(counterfactuals.shape[0]):
                        cf_list.append(counterfactuals.iloc[i:(i+1)])

                    counterfactual = cf_list
                    any_success = False
                    success_count = 0
                    exp_change_count = 0
                    total_nums = 0

                    # TODO: make each of these their own count for hits (store the full table not just the averaged results)
                    density_list = list()
                    distance_list = list()
                    for option in counterfactual:
                        #print(np.array(option[0]))
                        #print(np.array(input_row.iloc[0]))
                        #print(np.array(option[0]) != np.array(input_row.iloc[0]))
                        if len(option) > 0 and np.any(np.array(option) != np.array(initial_row.iloc[0])):
                            total_nums += 1
                            #input_row = option
                            #print(f"OPTION {total_nums}: {explainer.predict(input_row)[0]} vs {wanted_class}")
                            #if explainer.predict(input_row.to_numpy().reshape((1,-1)))[0] == wanted_class:
                            #    exp_change_count += 1
                            if bbmodel.predict(option.drop('target', axis=1, inplace=False))[0] == wanted_class:
                                success_count += 1
                                result_row.append(1)
                                any_success = True
                            else:
                                result_row.append(0)
                            distance_list.append(diff_calc.get_scaled_distance(pert_row.copy(), option))
                            density_list.append(diff_calc.get_density_distance(pert_row.copy(), option))

                        else:
                            result_row.append(-1)
                            density_list.append(-1)
                            distance_list.append(-1)

                    if len(counterfactual) < 5:
                        for _ in range(5-len(counterfactual)):
                            result_row.append(-1)
                            density_list.append(-1)
                            distance_list.append(-1)

                    if len(counterfactual) > 5:
                        result_row = result_row[:5]
                        density_list = density_list[:5]
                        distance_list = distance_list[:5]

                    result_row.append(int(any_success))
                    result_row += distance_list
                    result_row += density_list
                    result_row.append(success_count)

                results.loc[results.shape[0]] = result_row  # add to all records

                # delete to save space for next run
                #del explainer
                #del explanation

            #else:
            #    result_row += [-1, 0] + [-1 for _ in range(36)]
            #    results.loc[results.shape[0]] = result_row

    results.to_csv("Results/" + "_".join([data_name,
                                          "nruns" + str(local_test_end),
                                          "nbins" + str(n_bins),
                                          "nperturb" + str(n_perturbations),
                                          "devscalin" + str(dev_scaling)]) + "_counterfactuals_results.csv")


# TODO: NOTE: IRIS tests for LIME were BAD. It had 30% accuracy on the single/double blind
#  but these lowered as we considered the local bounds of the data. VAELime performs just as poorly.
# TODO: implement the VAE into BARBE for comparison...?
# TODO: double check that the data looks right for the model.
def distribution_experiment_iris():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names).astype(float)
    iris_df['target'] = iris.target
    iris_df = iris_df.sample(frac=1, random_state=117283).reset_index(drop=True)
    #test_euclidean_distance(iris_df, 0)
    #assert False

    split_point_test = int((iris_df.shape[0] * 0.2) // 1)  # 80-20 split
    iris_perturb = iris_df.iloc[0:split_point_test].drop('target', axis=1)
    iris_test = iris_df.iloc[0:split_point_test].drop('target', axis=1)
    iris_training = iris_df.iloc[split_point_test:].drop('target', axis=1)
    iris_training_label = iris['target'][split_point_test:]
    def class_fun(y):
        if y == 0:
            return "a"
        if y == 1:
            return "b"
        return "c"
    iris_training_label = [class_fun(y) for y in iris_training_label]
    '''
    lime_distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, dev_scaling=1,
                                 local_test_end=20, data_name="data", n_perturbations=100,
                                 lime_version=LimeNewPert, lime_discretizer='decile',
                                 use_barbe_perturbations=False):
    '''

    for pert_c in [1000]:#[100, 1000, 5000]:
        #lore_distribution_experiment(iris_training,
        #                             iris_training_label,
        #                             iris_perturb,
        #                             iris_test,
        #                             local_test_end=split_point_test,
        #                             data_name="neural_iris",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(iris_training,
        #                             iris_training_label,
         #                            iris_perturb,
         #                            iris_test,
         #                            lime_version=VAELimeNewPert,
         ##                            local_test_end=split_point_test,
          #                           data_name="vaelime_iris",
          #                           n_perturbations=pert_c,
          #                           use_barbe_perturbations=False,
          #                           dev_scaling=1)
        # lime_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=20,
        #                             data_name="vaelime_breast_cancer",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            for dev_n in [2, 3, 10, 50, 100]:
                distribution_experiment(iris_training,
                                         iris_training_label,
                                         iris_perturb,
                                         iris_test,
                                        # lime_version=LimeNewPert,
                                        local_test_end=split_point_test,
                                        data_name="barbe_neural_iris",
                                        n_perturbations=pert_c,
                                        # use_barbe_perturbations=False,
                                        n_bins=10,
                                        dev_scaling=dev_n)
                #lime_distribution_experiment(iris_training,
                #                             iris_training_label,
                #                             iris_perturb,
                #                             iris_test,
                #                             lime_version=VAELimeNewPert,
                #                             local_test_end=split_point_test,
                #                             data_name="vaelime_neural_iris",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=False,
                #                             dev_scaling=dev_n)


def distribution_experiment_breast_cancer():
    # set random seed
    breast_cancer = datasets.load_breast_cancer()
    breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    def class_fun(y):
        if 1 > y >= 0:
            return "a"
        if y >= 1:
            return "b"
        return "c"
    bctarget = [class_fun(y) for y in breast_cancer.target]
    breast_cancer_df['target'] = bctarget
    # randomize the order of the data
    breast_cancer_df = breast_cancer_df.sample(frac=1, random_state=117283).reset_index(drop=True)

    #print(breast_cancer_df)
    #assert False
    split_point_test = int((breast_cancer_df.shape[0] * 0.2) // 1)  # 80-20 split
    breast_cancer_perturb = breast_cancer_df.iloc[0:50].drop('target', axis=1)  # average over 50 cases to perturb
    breast_cancer_test = breast_cancer_df.iloc[0:split_point_test].drop('target', axis=1)
    breast_cancer_training = breast_cancer_df.iloc[split_point_test:].drop('target', axis=1)
    breast_cancer_training_label = breast_cancer_df['target'][split_point_test:]  # black box training labels


    for pert_c in [1000]:
        #lore_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             local_test_end=50,
        #                             data_name="lore_neural_breast_cancer",
        #                             n_perturbations=1000,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=50,
        ##                             data_name="vaelime_breast_cancer",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=20,
        #                             data_name="vaelime_breast_cancer",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            for dev_n in [1]:#[1, 2, 3, 10, 50, 100]:
                #distribution_experiment(breast_cancer_training,
                #                        breast_cancer_training_label,
                #                        breast_cancer_perturb,
                #                        breast_cancer_test,
                #                        #lime_version=LimeNewPert,
                #                        local_test_end=50,
                #                        data_name="barbe_neural_breast_cancer",
                #                        n_perturbations=pert_c,
                #                        #use_barbe_perturbations=False,
                #                        n_bins=10,
                #                        dev_scaling=dev_n)
                #lime_distribution_experiment(breast_cancer_training,
                #                             breast_cancer_training_label,
                #                             breast_cancer_perturb,
                #                             breast_cancer_test,
                #                             lime_version=LimeNewPert,
                #                             local_test_end=50,
                #                             data_name="bpert_neural_breast_cancer",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=True,
                #                             dev_scaling=dev_n)
                lime_distribution_experiment(breast_cancer_training,
                                             breast_cancer_training_label,
                                             breast_cancer_perturb,
                                             breast_cancer_test,
                                             lime_version=VAELimeNewPert,
                                             local_test_end=50,
                                             data_name="vaelime_neural_breast_cancer",
                                             n_perturbations=pert_c,
                                             use_barbe_perturbations=False,
                                             dev_scaling=dev_n)


from sklearn.model_selection import train_test_split


def counterfactual_experiment_breast_cancer(use_pretrained=False):
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    breast_cancer = datasets.load_breast_cancer()
    breast_cancer_df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    #print(breast_cancer_df.columns)
    #assert False
    def class_fun(y):
        if 1 > y >= 0:
            return "a"
        if y >= 1:
            return "b"
        return "c"

    #bctarget = [class_fun(y) for y in breast_cancer.target]
    #breast_cancer_df['target'] = breast_cancer.target
    # randomize the order of the data
    #breast_cancer_df = breast_cancer_df.sample(frac=1, random_state=117).reset_index(drop=True)
    breast_cancer_df, _, bc_target, _ = train_test_split(breast_cancer_df, breast_cancer.target, train_size=0.9999,
                                                   random_state=98147, shuffle=True) #117233


    y = [class_fun(y) for y in bc_target]
    data = breast_cancer_df
    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_part_train, loan_validate, loan_part_train_y, loan_validate_y = train_test_split(loan_training, loan_training_label, train_size=0.6, random_state=991246)
    n_estimators_options = [2, 5, 10, 20, 50]
    max_depth_options = [4, 10, 20, None]
    min_samples_options = [2, 10, 20, 40]
    options_list = [n_estimators_options, max_depth_options, min_samples_options]
    best_setting = (0, None)
    # TODO: add random forest too
    # for hidden_layer_setting in possible_hidden_layers:
    for n_estimators, max_depth, min_samples in product(*options_list):
        #encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
        #preprocess = ColumnTransformer([('enc', encoder, categorical_features)],
        #                               remainder='passthrough')
        model = Pipeline([#('pre', preprocess),
                          # ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer_setting,
                          #                      solver='adam',
                          #                      activation='relu',
                          #                      alpha=1e-8,
                          #                      tol=1e-16,
                          #                      max_iter=1000,
                          #                      random_state=301257))])
                          ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                                         max_depth=max_depth,
                                                         min_samples_split=min_samples,
                                                         min_samples_leaf=1,
                                                         bootstrap=True,
                                                         random_state=301257))])
        model.fit(loan_part_train, loan_part_train_y)
        curr_score = balanced_accuracy_score(loan_validate_y, model.predict(loan_validate))
        # print(hidden_layer_setting, ": ", curr_score)
        # print(hidden_layer_setting, ": ", confusion_matrix(attrition_part_train_y, model.predict(attrition_part_train)))
        if curr_score > best_setting[0]:
            best_setting = (curr_score, (n_estimators, max_depth, min_samples))

    n_estimators, max_depth, min_samples = best_setting[1]
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    #preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    model = Pipeline([#('pre', preprocess),
                      # ('clf', MLPClassifier(hidden_layer_sizes=best_setting[1],
                      #                      solver='adam',
                      #                          activation='relu',
                      #                          alpha=1e-8,
                      #                          tol=1e-16,
                      #                          max_iter=1000, random_state=301257)),
                      ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                                     max_depth=max_depth,
                                                     min_samples_split=min_samples,
                                                     min_samples_leaf=1,
                                                     bootstrap=True,
                                                     random_state=301257))
                      ])
    # ('clf', RandomForestClassifier(n_estimators=5,
    #                               max_depth=4,
    #                               min_samples_split=10,
    #                               min_samples_leaf=3,
    #                               bootstrap=True,
    #                               random_state=301257))])

    #model = FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=3)

    loan_training_stop = int(loan_training.shape[0] // 3)

    loan_perturb = loan_perturb.drop(['index'], axis=1)
    loan_test = loan_test.drop(['index'], axis=1)
    loan_training = loan_training.drop(['index'], axis=1)
    #print(loan_test)
    #assert False
    if not use_pretrained:
        model.fit(loan_training, loan_training_label)
        with open('../pretrained/bc_rf.pkl', 'wb') as f:
            dill.dump(model, f)
    else:
        with open('../pretrained/bc_rf.pkl', 'rb') as f:
            model = dill.load(f)

    print(confusion_matrix(loan_training_label, model.predict(loan_training)))
    #assert False

    for pert_c in [1000]:  # [100, 500, 1000]:
        #lore_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        #                             loan_test,
        #                             pre_trained_model=model,
        #                             local_test_end=50,
        #                             data_name="lore_neural_loan",
        #                             n_perturbations=1000,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        #                             loan_test,
        #                             pre_trained_model=model,
        #                             lime_version=LimeNewPert,
        #                             local_test_end=50,
        #                             data_name="original_neural_loan_acceptance",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        ##lime_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        ##                             loan_test,
        #                             pre_trained_model=model,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=50,
        #                             data_name="vaelime_loan",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            for dev_n in [1]:#[1, 2, 3, 10, 50, 100]:

                if True:
                    counterfactual_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            pre_trained_model=model,
                                            # lime_version=LimeNewPert,
                                            local_test_end=50,
                                            restricted_features=['mean radius', 'mean perimeter', 'mean area', 'mean compactness',
                                                                    'radius error', 'perimeter error', 'area error', 'compactness error'],
                                            data_name="barbe_cv_rf_breast_cancer",
                                            use_negation_rules=True,
                                            n_perturbations=pert_c,
                                           # use_barbe_perturbations=False,
                                            n_bins=5,
                                            dev_scaling=dev_n)
                    counterfactual_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            pre_trained_model=model,
                                            # lime_version=LimeNewPert,
                                            local_test_end=50,
                                            restricted_features=['mean radius', 'mean perimeter', 'mean area', 'mean compactness',
                                                                    'radius error', 'perimeter error', 'area error', 'compactness error'],
                                            data_name="barbe_no_negation_cv_rf_breast_cancer",
                                            use_negation_rules=False,
                                            n_perturbations=pert_c,
                                           # use_barbe_perturbations=False,
                                            n_bins=5,
                                            dev_scaling=dev_n)
                    lore_counterfactual_experiment(loan_training.copy(),
                                                   loan_training_label.copy(),
                                                   loan_perturb.copy(),
                                                   loan_test.copy(),
                                                   pre_trained_model=model,
                                                   # lime_version=LimeNewPert,
                                                   local_test_end=50,
                                                   data_name="lore_cv_rf_breast_cancer",
                                                   restricted_features=['mean radius', 'mean perimeter', 'mean area', 'mean compactness',
                                                                    'radius error', 'perimeter error', 'area error', 'compactness error'],
                                                   n_perturbations=pert_c,
                                                   # use_barbe_perturbations=False,
                                                   n_bins=5,
                                                   dev_scaling=dev_n)
                dice_counterfactual_experiment(loan_training.copy(),
                                               loan_training_label.copy(),
                                               loan_perturb.copy(),
                                               loan_test.copy(),
                                               pre_trained_model=model,
                                               discrete_features=[],
                                               # lime_version=LimeNewPert,
                                               local_test_end=50,
                                               data_name="dice_cv_rf_breast_cancer",
                                               restricted_features=['mean radius', 'mean perimeter', 'mean area', 'mean compactness',
                                                                    'radius error', 'perimeter error', 'area error', 'compactness error'],
                                               n_perturbations=pert_c,
                                               # use_barbe_perturbations=False,
                                               n_bins=5,
                                               dev_scaling=dev_n)
                #lime_distribution_experiment(loan_training,
                ##                             loan_training_label,
                 #                            loan_perturb,
                 #                            loan_test,
                 #                            pre_trained_model=model,
                 #                            lime_version=LimeNewPert,
                 #                            local_test_end=50,
                 #                            data_name="bpert_cats_neural_loan_acceptance",
                 #                            n_perturbations=pert_c,
                 #                            use_barbe_perturbations=True,
                 #                            dev_scaling=dev_n)
                ##lime_distribution_experiment(loan_training,
                 #                            loan_training_label,
                 ##                            loan_perturb,
                  #                           loan_test,
                  #                           pre_trained_model=model,
                  #                           lime_version=VAELimeNewPert,
                  #                           local_test_end=50,
                   #                          data_name="vaelime_cats_neural_loan_acceptance",
                   #                          n_perturbations=pert_c,
                   #                          use_barbe_perturbations=False,
                   #                          dev_scaling=dev_n)


def counterfactual_experiment_loan(use_pretrained=False):
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/train_loan_raw.csv")
    data = data.drop('Loan_ID', axis=1)
    print(list(data))
    data = data.dropna()
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    data = data.dropna()
    data = data.sample(frac=1, random_state=78121)
    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Loan_Status']:
            data[cat] = data[cat].astype(float)

    y = data['Loan_Status']
    data = data.drop(['Loan_Status'], axis=1)
    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_part_train, loan_validate, loan_part_train_y, loan_validate_y = train_test_split(loan_training,
                                                                                          loan_training_label,
                                                                                          train_size=0.6,
                                                                                          random_state=991246)
    n_estimators_options = [2, 5, 10, 20, 50]
    max_depth_options = [4, 10, 20, None]
    min_samples_options = [2, 10, 20, 40]
    options_list = [n_estimators_options, max_depth_options, min_samples_options]
    best_setting = (0, None)
    # TODO: add random forest too
    # for hidden_layer_setting in possible_hidden_layers:
    for n_estimators, max_depth, min_samples in product(*options_list):
        encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
        preprocess = ColumnTransformer([('enc', encoder, categorical_features)],
                                       remainder='passthrough')
        model = Pipeline([  ('pre', preprocess),
            # ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer_setting,
            #                      solver='adam',
            #                      activation='relu',
            #                      alpha=1e-8,
            #                      tol=1e-16,
            #                      max_iter=1000,
            #                      random_state=301257))])
            ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples,
                                           min_samples_leaf=1,
                                           bootstrap=True,
                                           random_state=301257))])
        model.fit(loan_part_train, loan_part_train_y)
        curr_score = balanced_accuracy_score(loan_validate_y, model.predict(loan_validate))
        # print(hidden_layer_setting, ": ", curr_score)
        # print(hidden_layer_setting, ": ", confusion_matrix(attrition_part_train_y, model.predict(attrition_part_train)))
        if curr_score > best_setting[0]:
            best_setting = (curr_score, (n_estimators, max_depth, min_samples))

    n_estimators, max_depth, min_samples = best_setting[1]
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    model = Pipeline([  ('pre', preprocess),
        # ('clf', MLPClassifier(hidden_layer_sizes=best_setting[1],
        #                      solver='adam',
        #                          activation='relu',
        #                          alpha=1e-8,
        #                          tol=1e-16,
        #                          max_iter=1000, random_state=301257)),
        ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples,
                                       min_samples_leaf=1,
                                       bootstrap=True,
                                       random_state=301257))
    ])

    #model = FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=3)

    loan_training_stop = int(loan_training.shape[0] // 3)

    discrete_features = list()
    for feature in list(loan_training):
        unique_values = np.unique(loan_training.iloc[0:loan_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
            discrete_features.append(feature)
            loan_perturb[feature] = [(value if value in unique_values else "unknown") for value in loan_perturb[feature]]
            loan_test[feature] = [(value if value in unique_values else "unknown") for value in loan_test[feature]]
            loan_training[feature] = [(value if value in unique_values else "unknown") for value in loan_training[feature]]
            #print(np.unique(loan_training[feature]))
            #print(np.unique(loan_test[feature]))

    loan_perturb = loan_perturb.drop(['index'], axis=1)
    loan_test = loan_test.drop(['index'], axis=1)
    loan_training = loan_training.drop(['index'], axis=1)
    #print(loan_test)
    #assert False
    if not use_pretrained:
        model.fit(loan_training, loan_training_label)
        with open('../pretrained/loan_rf.pkl', 'wb') as f:
            dill.dump(model, f)
    else:
        with open('../pretrained/loan_rf.pkl', 'rb') as f:
            model = dill.load(f)

    #print(confusion_matrix(loan_training_label, model.predict(loan_training)))
    #assert False

    for pert_c in [1000]:  # [100, 500, 1000]:
        #lore_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        #                             loan_test,
        #                             pre_trained_model=model,
        #                             local_test_end=50,
        #                             data_name="lore_neural_loan",
        #                             n_perturbations=1000,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        #                             loan_test,
        #                             pre_trained_model=model,
        #                             lime_version=LimeNewPert,
        #                             local_test_end=50,
        #                             data_name="original_neural_loan_acceptance",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        ##lime_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        ##                             loan_test,
        #                             pre_trained_model=model,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=50,
        #                             data_name="vaelime_loan",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            for dev_n in [1]:#[1, 2, 3, 10, 50, 100]:

                if True:
                    counterfactual_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            pre_trained_model=model,
                                            # lime_version=LimeNewPert,
                                            local_test_end=50,
                                            restricted_features=['Gender', 'Married', 'Dependents'],
                                            data_name="barbe_cv_rf_loan_acceptance",
                                            use_negation_rules=True,
                                            n_perturbations=pert_c,
                                           # use_barbe_perturbations=False,
                                            n_bins=10,
                                            dev_scaling=dev_n)
                    counterfactual_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            pre_trained_model=model,
                                            # lime_version=LimeNewPert,
                                            local_test_end=50,
                                            restricted_features=['Gender', 'Married', 'Dependents'],
                                            data_name="barbe_no_negation_cv_rf_loan_acceptance",
                                            use_negation_rules=False,
                                            n_perturbations=pert_c,
                                           # use_barbe_perturbations=False,
                                            n_bins=10,
                                            dev_scaling=dev_n)
                    lore_counterfactual_experiment(loan_training.copy(),
                                                   loan_training_label.copy(),
                                                   loan_perturb.copy(),
                                                   loan_test.copy(),
                                                   pre_trained_model=model,
                                                   # lime_version=LimeNewPert,
                                                   local_test_end=50,
                                                   data_name="lore_cv_rf_loan_acceptance",
                                                   restricted_features=['Gender', 'Married', 'Dependents'],
                                                   n_perturbations=pert_c,
                                                   # use_barbe_perturbations=False,
                                                   n_bins=10,
                                                   dev_scaling=dev_n)
                dice_counterfactual_experiment(loan_training.copy(),
                                               loan_training_label.copy(),
                                               loan_perturb.copy(),
                                               loan_test.copy(),
                                               pre_trained_model=model,
                                               discrete_features=categorical_features + ['Loan_Amount_Term'],
                                               # lime_version=LimeNewPert,
                                               local_test_end=50,
                                               data_name="dice_cv_rf_loan_acceptance",
                                               restricted_features=['Gender', 'Married', 'Dependents'],
                                               n_perturbations=pert_c,
                                               # use_barbe_perturbations=False,
                                               n_bins=10,
                                               dev_scaling=dev_n)
                #lime_distribution_experiment(loan_training,
                ##                             loan_training_label,
                 #                            loan_perturb,
                 #                            loan_test,
                 #                            pre_trained_model=model,
                 #                            lime_version=LimeNewPert,
                 #                            local_test_end=50,
                 #                            data_name="bpert_cats_neural_loan_acceptance",
                 #                            n_perturbations=pert_c,
                 #                            use_barbe_perturbations=True,
                 #                            dev_scaling=dev_n)
                ##lime_distribution_experiment(loan_training,
                 #                            loan_training_label,
                 ##                            loan_perturb,
                  #                           loan_test,
                  #                           pre_trained_model=model,
                  #                           lime_version=VAELimeNewPert,
                  #                           local_test_end=50,
                   #                          data_name="vaelime_cats_neural_loan_acceptance",
                   #                          n_perturbations=pert_c,
                   #                          use_barbe_perturbations=False,
                   #                          dev_scaling=dev_n)


def counterfactual_experiment_attrition(use_pretrained=False):
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/ibm_hr_attrition.csv")
    data = data.drop(['StandardHours', 'EmployeeCount', 'Over18'], axis=1)
    print(list(data))
    data = data.dropna()
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    data = data.dropna()
    data = data.sample(frac=1, random_state=5771)
    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Attrition']:
            data[cat] = data[cat].astype(float)

    y = data['Attrition']
    data = data.drop(['Attrition'], axis=1)

    #preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    #model = Pipeline([('pre', CategoricalEncoder()),
    #                  ('clf', FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=2))])
    #('clf', MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257))])
                      #('clf', RandomForestClassifier(n_estimators=5,
                      #                               max_depth=4,
                      #                               min_samples_split=10,
                      #                               min_samples_leaf=3,
                      #                               bootstrap=True,
                      #                               random_state=301257))])

    #model = FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=4)
    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    attrition_perturb = data.iloc[7:57].reset_index()
    attrition_test = data.iloc[0:split_point_test].reset_index()
    attrition_training = data.iloc[split_point_test:].reset_index()
    attrition_training_label = y[split_point_test:]

    attrition_part_train, attrition_validate, attrition_part_train_y, attrition_validate_y = train_test_split(attrition_training, attrition_training_label, train_size=0.7, random_state=991246)
    #possible_hidden_layers = [(50, 50, 20, 5,), (50, 50, 10), (50, 50, 5), (50, 20, 2), (50, 10, 5), (20, 10, 5), (10, 10, 5),
    #                          (10, 5), (5, 5), (200,), (100,), (50,), (20,), (10,), (5,), (2,)]
    n_estimators_options = [2, 5, 10, 20, 50]
    max_depth_options = [4, 10, 20, None]
    min_samples_options = [2, 10, 20, 40]
    options_list = [n_estimators_options, max_depth_options, min_samples_options]
    best_setting = (0, None)
    # TODO: add random forest too
    #for hidden_layer_setting in possible_hidden_layers:
    for n_estimators, max_depth, min_samples in product(*options_list):
        encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
        preprocess = ColumnTransformer([('enc', encoder, categorical_features)],
                                       remainder='passthrough')
        model = Pipeline([('pre', preprocess),
                          #('clf', MLPClassifier(hidden_layer_sizes=hidden_layer_setting,
                          #                      solver='adam',
                          #                      activation='relu',
                          #                      alpha=1e-8,
                          #                      tol=1e-16,
                          #                      max_iter=1000,
                          #                      random_state=301257))])
                          ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                                         max_depth=max_depth,
                                                         min_samples_split=min_samples,
                                                         min_samples_leaf=1,
                                                         bootstrap=True,
                                                         random_state=301257))])
        model.fit(attrition_part_train, attrition_part_train_y)
        curr_score = balanced_accuracy_score(attrition_validate_y, model.predict(attrition_validate))
        #print(hidden_layer_setting, ": ", curr_score)
        #print(hidden_layer_setting, ": ", confusion_matrix(attrition_part_train_y, model.predict(attrition_part_train)))
        if curr_score > best_setting[0]:
            best_setting = (curr_score, (n_estimators, max_depth, min_samples))

    n_estimators, max_depth, min_samples = best_setting[1]
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    model = Pipeline([('pre', preprocess),
                      #('clf', MLPClassifier(hidden_layer_sizes=best_setting[1],
                      #                      solver='adam',
                      #                          activation='relu',
                      #                          alpha=1e-8,
                      #                          tol=1e-16,
                      #                          max_iter=1000, random_state=301257)),
                      ('clf', RandomForestClassifier(n_estimators=n_estimators,
                                                     max_depth=max_depth,
                                                     min_samples_split=min_samples,
                                                     min_samples_leaf=1,
                                                     bootstrap=True,
                                                     random_state=301257))
                       ])

    attrition_training_stop = int(attrition_training.shape[0] // 3)

    for feature in list(attrition_training):
        unique_values = np.unique(attrition_training.iloc[0:attrition_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
            print(unique_values)
            attrition_perturb[feature] = [(value if value in unique_values else "unknown") for value in attrition_perturb[feature]]
            attrition_test[feature] = [(value if value in unique_values else "unknown") for value in attrition_test[feature]]
            attrition_training[feature] = [(value if value in unique_values else "unknown") for value in attrition_training[feature]]
            #print(np.unique(loan_training[feature]))
            #print(np.unique(loan_test[feature]))

    attrition_perturb = attrition_perturb.drop(['index'], axis=1)
    attrition_test = attrition_test.drop(['index'], axis=1)
    attrition_training = attrition_training.drop(['index'], axis=1)
    #print(loan_test)
    #assert False
    if not use_pretrained:
        model.fit(attrition_training, attrition_training_label)
        with open('../pretrained/attrition_rf.pkl', 'wb') as f:
            dill.dump(model, f)
    else:
        with open('../pretrained/attrition_rf.pkl', 'rb') as f:
            model = dill.load(f)
    print(confusion_matrix(attrition_training_label, model.predict(attrition_training)))
    #assert False
    print(model.predict(attrition_perturb))
    #assert False

    for pert_c in [1000]:  # [100, 500, 1000]:
        #lore_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        #                             loan_test,
        #                             pre_trained_model=model,
        #                             local_test_end=50,
        #                             data_name="lore_neural_loan",
        #                             n_perturbations=1000,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        #                             loan_test,
        #                             pre_trained_model=model,
        #                             lime_version=LimeNewPert,
        #                             local_test_end=50,
        #                             data_name="original_neural_loan_acceptance",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        ##lime_distribution_experiment(loan_training,
        #                             loan_training_label,
        #                             loan_perturb,
        ##                             loan_test,
        #                             pre_trained_model=model,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=50,
        #                             data_name="vaelime_loan",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            for dev_n in [1]:#[1, 2, 3, 10, 50, 100]:
                counterfactual_experiment(attrition_training,
                                        attrition_training_label,
                                        attrition_perturb,
                                        attrition_test,
                                        pre_trained_model=model,
                                        # lime_version=LimeNewPert,
                                        local_test_end=50,
                                        data_name="barbe_cv_rf_attrition",
                                        n_perturbations=pert_c,
                                       # use_barbe_perturbations=False,
                                          use_negation_rules=True,
                                        restricted_features=['Gender', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'],
                                        n_bins=5,
                                        dev_scaling=dev_n)
                counterfactual_experiment(attrition_training,
                                        attrition_training_label,
                                        attrition_perturb,
                                        attrition_test,
                                        pre_trained_model=model,
                                        # lime_version=LimeNewPert,
                                        local_test_end=50,
                                        data_name="barbe_no_negation_cv_rf_attrition",
                                        n_perturbations=pert_c,
                                       # use_barbe_perturbations=False,
                                          use_negation_rules=False,
                                        restricted_features=['Gender', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'],
                                        n_bins=5,
                                        dev_scaling=dev_n)
                dice_counterfactual_experiment(attrition_training,
                                        attrition_training_label,
                                        attrition_perturb,
                                        attrition_test,
                                        pre_trained_model=model,
                                        # lime_version=LimeNewPert,
                                        discrete_features=categorical_features,
                                        local_test_end=50,
                                        data_name="dice_cv_rf_attrition",
                                        n_perturbations=pert_c,
                                        restricted_features=['Gender', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'],
                                       # use_barbe_perturbations=False,
                                        n_bins=5,
                                        dev_scaling=dev_n)
                lore_counterfactual_experiment(attrition_training,
                                        attrition_training_label,
                                        attrition_perturb,
                                        attrition_test,
                                        pre_trained_model=model,
                                        # lime_version=LimeNewPert,
                                        local_test_end=50,
                                        data_name="lore_cv_rf_attrition",
                                        n_perturbations=pert_c,
                                        restricted_features=['Gender', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'],
                                       # use_barbe_perturbations=False,
                                        n_bins=5,
                                        dev_scaling=dev_n)
                #lime_distribution_experiment(loan_training,
                ##                             loan_training_label,
                 #                            loan_perturb,
                 #                            loan_test,
                 #                            pre_trained_model=model,
                 #                            lime_version=LimeNewPert,
                 #                            local_test_end=50,
                 #                            data_name="bpert_cats_neural_loan_acceptance",
                 #                            n_perturbations=pert_c,
                 #                            use_barbe_perturbations=True,
                 #                            dev_scaling=dev_n)
                ##lime_distribution_experiment(loan_training,
                 #                            loan_training_label,
                 ##                            loan_perturb,
                  #                           loan_test,
                  #                           pre_trained_model=model,
                  #                           lime_version=VAELimeNewPert,
                  #                           local_test_end=50,
                   #                          data_name="vaelime_cats_neural_loan_acceptance",
                   #                          n_perturbations=pert_c,
                   #                          use_barbe_perturbations=False,
                   #                          dev_scaling=dev_n)


def distribution_experiment_libras():
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/movement_libras.data",
                       names=([str(i) for i in range(1, 91)]+['target']),
                       index_col=False)
    print(data)
    print(data.shape)
    print(list(data))
    data = data[[str(i) for i in range(1, 91, 2)]+['target']]
    data = data.sample(frac=1, random_state=117283)
    data.reset_index(drop=True, inplace=True)
    libtarget = [chr(y + 96) for y in data['target']]
    y = libtarget# data['target'].astype(str)
    data = data.drop(['target'], axis=1)


    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    libras_perturb = data.iloc[0:30]
    libras_test = data.iloc[0:split_point_test]
    libras_training = data.iloc[split_point_test:]
    libras_training_label = y[split_point_test:]

    #print(libras_test)
    #assert False

    for pert_c in [1000]: # [100, 500, 1000]:
        #lore_distribution_experiment(libras_training,
        #                             libras_training_label,
        #                             libras_perturb,
        #                             libras_test,
        #                             local_test_end=30,
        #                             data_name="lore_neural_goodfit_libras",
        #                             n_perturbations=1000,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        #lime_distribution_experiment(libras_training,
        #                             libras_training_label,
        #                             libras_perturb,
        #                             libras_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=30,
        #                             data_name="vaelime_libras",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        # lime_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=20,
        #                             data_name="vaelime_breast_cancer",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            for dev_n in [1]:# [1, 2, 3, 10, 50, 100]:
                #lime_distribution_experiment(libras_training,
                #                             libras_training_label,
                #                             libras_perturb,
                #                             libras_test,
                #                             lime_version=LimeNewPert,
                #                             local_test_end=30,
                #                             data_name="bpert_neural_goodfit_libras",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=True,
                #                             dev_scaling=dev_n)
                lime_distribution_experiment(libras_training,
                                             libras_training_label,
                                             libras_perturb,
                                             libras_test,
                                             lime_version=VAELimeNewPert,
                                             local_test_end=30,
                                             data_name="vaelime_neural_goodfit_libras",
                                             n_perturbations=pert_c,
                                             use_barbe_perturbations=False,
                                             dev_scaling=dev_n)
                #distribution_experiment(libras_training,
                #                        libras_training_label,
                #                        libras_perturb,
                ##                        libras_test,
                #                        # pre_trained_model=model,
                #                        # lime_version=LimeNewPert,
                #                        local_test_end=30,
                #                        data_name="barbe_neural_goodfit_libras",
                #                        n_perturbations=pert_c,
                #                        # use_barbe_perturbations=False,
                ##                        n_bins=5,
                 #                       dev_scaling=dev_n)


#test_distribution_experiment_iris()
if __name__ == '__main__':
    #simple_distribution_experiment_simulated()
    counterfactual_experiment_loan(use_pretrained=False)
    #counterfactual_experiment_loan(use_pretrained=True)
    #counterfactual_experiment_attrition(use_pretrained=False)


# TODO: add a timeseries dataset
# TODO: test on more datasets with >2 labelss
# TODO: add tests with neural networks (should pretrain for all of them)
# TODO: add perturbation timing tests
# TODO: add tests with LORE
#       - LORE may perform better but take more time so we need to communicate this
#       - what exactly does lore need too??
# TODO: get VAE-LIME working (and run all previous tests)
#       - point out that a VAE needs lots of data to perform well, which we may not have
# TODO: tally failures for each method
#       - count LIME failures when the class is not correct in the perturbed sample
#       - this is the hit rate which is used by others
# TODO: add perturbations to the acquired features (scale/std and covariance) to
#       see how the resulting performance is impacted by erroneous features

# TODO: table structure
#  Dataset, Eval, LIME, LIME+US, LIME+VAE, BARBE, BARBE+US, LORE, LORE+US || PROB. LIME
#  Baseline Guesser (guess the same class as input for every value)
#  BC, Perturbation Fidelity, 0.8 +/- 0.1, 0.9, 0.7, 0.99, ...
#  BC, Global Single-Blind
#  BC, Local Single-Blind
#  BC, Global Double-Blind
#  BC, Local Double-Blind
#  BC, Hit Rate
#  ...
#  BC Average, Fidelity
