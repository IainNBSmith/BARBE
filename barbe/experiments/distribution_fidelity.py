# IAIN make this file get and store results for each of the fidelity values we will check
#  against each other for each distribution type + lime1's distribution

# experiments to see which of Lime, Uniform, Standard-Normal, Multi-Normal, Clustered-Normal, t-Distribution, Chauchy
#  perform the best in terms of fidelity values

# pert_fidelity = barbe[i].fidelity() -> training accuracy
# train_fidelity = barbe[i].fidelity(training_data, bbmodel) -> single blind / validation accuracy
# test_fidelity = barbe[i].fidelity(bbmodel_data, bbmodel) -> double blind / testing accuracy

from barbe.utils.lime_interface import LimeNewPert, VAELimeNewPert
from barbe.discretizer import CategoricalEncoder
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics import accuracy_score
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


def test_euclidean_distance(training_data, index):
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


def distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                            discrete_features=None,
                            local_test_end=20, data_name="data", n_perturbations=100, n_bins=5, dev_scaling=10):
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

    #barbe_dist = ['standard-normal', 'normal', 'uniform', 'cauchy', 't-distribution']
    barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end)]
    fidelity_single_blind = [[] for _ in range(local_test_end)]
    fidelity_double_blind = [[] for _ in range(local_test_end)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end)]

    # NEW
    fidelity_single_diff_n = [[] for _ in range(local_test_end)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end)]

    hit_rate = [[] for _ in range(local_test_end)]

    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:

            try:
                explainer = BARBE(training_data=iris_wb_training,
                                  input_bounds=None,#[(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                                  perturbation_type=distribution,
                                  n_perturbations=n_perturbations,
                                  dev_scaling_factor=dev_scaling,
                                  n_bins=n_bins,
                                  verbose=False,
                                  input_sets_class=False)

                explanation = explainer.explain(pert_row, bbmodel)
                #print(explanation)
                #print("IAIN DATA: ", explainer._perturbed_data)
                #assert False
                weight = None  # or nearest or euclidean

                temp_pert = explainer.get_surrogate_fidelity()
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
                # NEW
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
                # NEW
                fidelity_single_diff_n[i].append(temp_single - temp_single_f)
                fidelity_double_diff_n[i].append(temp_double - temp_double_f)
                hit_rate[i].append(1)

                del explainer
                del explanation

            except:
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
                # NEW
                fidelity_single_diff_n[i].append(None)
                fidelity_single_diff_e[i].append(None)
                fidelity_double_diff_n[i].append(None)
                fidelity_double_diff_e[i].append(None)
                hit_rate[i].append(0)

            #print(i, " - ", distribution)
            #print(temp_pert)
            #print(temp_single)
            #print(temp_double)


    # NEW
    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate"]]
    #print(fidelity_pert)
    #print(fidelity_single_blind)
    #print(fidelity_double_blind)
    for fidelity_pert, fidelity_single_blind, fidelity_double_blind, run in \
        [(fidelity_pert, fidelity_single_blind, fidelity_double_blind, 'regular'),
         (fidelity_pert, fidelity_single_blind_e, fidelity_double_blind_e, 'euclidean'),
         (fidelity_pert, fidelity_single_blind_n, fidelity_double_blind_n, 'nearest neighbors'),
         # NEW
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
                    temp_pert_std += (fidelity_pert[i][j] - temp_pert_acc/mean_count)**2
                    temp_single_std += (fidelity_single_blind[i][j] - temp_single_acc/mean_count)**2
                    temp_double_std += (fidelity_double_blind[i][j] - temp_double_acc/mean_count)**2
            temp_pert_std = (temp_pert_std/(mean_count-1))
            temp_single_std = (temp_single_std / (mean_count - 1))
            temp_double_std = (temp_double_std / (mean_count - 1))
            #print(barbe_dist[j])
            if mean_count != 0:
                # add standard deviations as info
                if (len(averages_print)-1)/3 <= j:
                    averages_print.append([barbe_dist[j], "Perturbed"])
                    averages_print.append([barbe_dist[j], "Single Blind"])
                    averages_print.append([barbe_dist[j], "Double Blind"])
                averages_print[(j*3)+1].append(temp_pert_acc/mean_count)
                averages_print[(j * 3) + 1].append(temp_pert_std)
                averages_print[(j*3)+2].append(temp_single_acc/mean_count)
                averages_print[(j * 3) + 2].append(temp_single_std)
                averages_print[(j*3)+3].append(temp_double_acc/mean_count)
                averages_print[(j * 3) + 3].append(temp_double_std)
    for j in range(len(barbe_dist)):
        average_hits = 0
        for i in range(local_test_end):
            average_hits += hit_rate[i][j]
        average_hits /= local_test_end
        averages_print[(j * 3) + 1].append(average_hits)
        averages_print[(j * 3) + 2].append(0)
        averages_print[(j * 3) + 3].append(0)
                #print("Comparison Measure: ", run)
                #print("Fidelity: ", temp_pert_acc/mean_count)
                #print("LIME: ", np.mean(lime_fidelity_pert))
                #print("Single Blind: ", temp_single_acc/mean_count)
                #print("LIME: ", np.mean(lime_fidelity_single_blind))
                #print("Double Blind: ", temp_double_acc/mean_count)
                #print("LIME: ", np.mean(lime_fidelity_double_blind))

    #print(averages_print)
    #for s in averages_print:
    #    print(*s)

    pd.DataFrame(averages_print).to_csv("Results/"+"_".join([data_name,
                                                             "nruns"+str(local_test_end),
                                                             "nbins"+str(n_bins),
                                                             "nperturb"+str(n_perturbations),
                                                             "devscalin"+str(dev_scaling)])+"_results.csv")
    #print([(np.nanmin(iris_numpy[:,0]), np.nanmax(iris_numpy[:,0])),
    #                                            (np.nanmin(iris_numpy[:,1]), np.nanmax(iris_numpy[:,1])),
    #                                            (np.nanmin(iris_numpy[:,2]), np.nanmax(iris_numpy[:,2])),
    #                                            (np.nanmin(iris_numpy[:,3]), np.nanmax(iris_numpy[:,3])),])


# TODO: NOTE: IRIS tests for LIME were BAD. It had 30% accuracy on the single/double blind
#  but these lowered as we considered the local bounds of the data. VAELime performs just as poorly.
# TODO: implement the VAE into BARBE for comparison...?
# TODO: double check that the data looks right for the model.
def test_distribution_experiment_iris():
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
        lore_distribution_experiment(iris_training,
                                     iris_training_label,
                                     iris_perturb,
                                     iris_test,
                                     local_test_end=split_point_test,
                                     data_name="neural_iris",
                                     n_perturbations=pert_c,
                                     use_barbe_perturbations=False,
                                     dev_scaling=1)
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


def test_distribution_experiment_breast_cancer():
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


def test_simple_distribution_experiment_simulated():
    X, y, z = simulate_linear_classified(seed_num=52146, size=1000, clusters_keep=3)
    X2, y2, _ = simulate_linear_classified(seed_num=89213, size=1000, clusters_keep=1)
    X3, y3, _ = simulate_linear_classified(seed_num=78761, size=1000, clusters_keep=2)
    X4, _, z4 = simulate_linear_classified(seed_num=11487, size=250, clusters_keep=3)
    y2_sort = y2.argsort()
    y3_sort = y3.argsort()
    print(X)
    print(y)
    print(sum(y))
    X = pd.DataFrame(X, columns=[str(i) for i in range(1, 4+1)])
    X2 = pd.DataFrame(X2, columns=[str(i) for i in range(1, 4+1)])
    X3 = pd.DataFrame(X3, columns=[str(i) for i in range(1, 4+1)])
    X4 = pd.DataFrame(X4, columns=[str(i) for i in range(1, 4+1)])
    # rounded linear regression performs poorly
    # a decision tree is surprisingly well suited to this problem (almost perfect)
    # SVC with rbf kernel performs ok
    # MLP performs very well when alpha is high (1e-3)
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-3,
                        hidden_layer_sizes=(8, 3, 3),
                        random_state=678993)
    # TODO: train a post-hoc explainer and compare where misclassifications occur
    #  plot the point that is being classified and circle the regions of percentages in each dimension
    #  plot the perturbed data and compare
    # clf = DecisionTreeClassifier()
    exp = BARBE(training_data=X,
                input_bounds=None,#[(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                perturbation_type='t-distribution',
                n_perturbations=5000,
                dev_scaling_factor=3,
                n_bins=10,
                verbose=False,
                input_sets_class=False)
    clf.fit(X, y)
    class_point = 999
    exp.explain(X2.iloc[class_point], clf)
    Xp = exp.get_perturbed_data()
    #print(Xp)
    #print(exp.get_surrogate_fidelity())
    #assert False

    y_pred = np.round(clf.predict(X))
    y2_pred = np.round(clf.predict(X2))
    y3_pred = np.round(clf.predict(X3))
    yp_pred = np.round(clf.predict(Xp))
    y4_pred = np.round(clf.predict(X4))
    y4_exp = exp.predict(X4)
    # show the quality of fit and difficulty on this data
    print("Evaluation for X")
    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print("Evaluation for Group 1")
    print(accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))
    print("Evaluation for Group 2")
    print(accuracy_score(y3, y3_pred))
    print(confusion_matrix(y3, y3_pred))
    print("Evaluation Compared to Perturber")
    print(accuracy_score(exp._blackbox_classification['perturbed'], exp._surrogate_classification['perturbed']))
    print(confusion_matrix(exp._blackbox_classification['perturbed'], exp._surrogate_classification['perturbed']))
    print("Evaluation Compared to Perturber on Holdout")
    print(accuracy_score(y4_pred, y4_exp.astype(int)))
    print(confusion_matrix(y4_pred, y4_exp.astype(int)))
    X = X.to_numpy()
    X2 = X2.to_numpy()
    X3 = X3.to_numpy()
    Xp = Xp.to_numpy()
    X4 = X4.to_numpy()

    # plot the results by class and cluster
    color_list = np.empty(shape=y.shape, dtype=np.str_)
    color_list[(y == 1) & (z == 1)] = 'red'
    color_list[(y == 0) & (z == 1)] = 'blue'
    color_list[(y == 1) & (z == 2)] = 'magenta'
    color_list[(y == 0) & (z == 2)] = 'cyan'
    fig = plt.figure(1)
    for i in range(4):
        for j in range(4):
            fig.add_subplot(4, 4, 4 * i + j + 1)
            if i >= j:
                plt.scatter(X[:, i], X[:, j], c=color_list)
            else:
                plt.scatter(X[-1:0:-1, i], X[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=yp_pred.shape, dtype=np.str_)
    print(exp._surrogate_classification['perturbed'])
    color_list[(yp_pred == 1) & (exp._surrogate_classification['perturbed'] == '1')] = 'r'
    color_list[(yp_pred == 0) & (exp._surrogate_classification['perturbed'] == '0')] = 'b'
    color_list[(yp_pred == 1) & (exp._surrogate_classification['perturbed'] == '0')] = 'y'
    color_list[(yp_pred == 0) & (exp._surrogate_classification['perturbed'] == '1')] = 'k'
    color_list[0] = 'g'
    #color_list[(y == 1) & (z == 2)] = 'magenta'
    #color_list[(y == 0) & (z == 2)] = 'cyan'

    #pca = PCA(n_components=4)
    #X_pca = pca.fit_transform(X)
    fig = plt.figure(2)
    for i in range(4):
        for j in range(4):
            fig.add_subplot(4, 4, 4 * i + j + 1)
            if i >= j:
                plt.scatter(Xp[color_list == 'r', i], Xp[color_list == 'r', j], c='r')
                plt.scatter(Xp[color_list == 'b', i], Xp[color_list == 'b', j], c='b')
                plt.scatter(Xp[color_list == 'y', i], Xp[color_list == 'y', j], c='yellow')
                plt.scatter(Xp[color_list == 'k', i], Xp[color_list == 'k', j], c='k')
            else:
                #plt.scatter(Xp[-1:0:-1, i], Xp[-1:0:-1, j], c=color_list[-1:0:-1])
                plt.scatter(Xp[color_list == 'k', i], Xp[color_list == 'k', j], c='k')
                plt.scatter(Xp[color_list == 'y', i], Xp[color_list == 'y', j], c='y')
                plt.scatter(Xp[color_list == 'b', i], Xp[color_list == 'b', j], c='b')
                plt.scatter(Xp[color_list == 'r', i], Xp[color_list == 'r', j], c='r')

            plt.scatter(Xp[class_point, i], Xp[class_point, j], c='g')
            plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y.shape, dtype=np.str_)
    color_list[(y_pred == 1) & (z == 1)] = 'red'
    color_list[(y_pred == 0) & (z == 1)] = 'blue'
    color_list[(y_pred == 1) & (z == 2)] = 'magenta'
    color_list[(y_pred == 0) & (z == 2)] = 'cyan'
    fig = plt.figure(3)
    for i in range(4):
        for j in range(4):
            fig.add_subplot(4, 4, 4 * i + j + 1)
            if i >= j:
                plt.scatter(X[:, i], X[:, j], c=color_list)

            else:
                plt.scatter(X[-1:0:-1, i], X[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y2.shape, dtype=np.str_)
    color_list[(y2_pred == 1) & (y2 == 1)] = 'red'
    color_list[(y2_pred == 0) & (y2 == 0)] = 'blue'
    color_list[(y2_pred == 1) & (y2 == 0)] = 'yellow'
    color_list[(y2_pred == 0) & (y2 == 1)] = 'green'
    fig = plt.figure(4)
    for i in range(4):
        for j in range(4):
            fig.add_subplot(4, 4, 4 * i + j + 1)
            if i >= j:
                plt.scatter(X2[:, i], X2[:, j], c=color_list)
            else:
                plt.scatter(X2[-1:0:-1, i], X2[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y3.shape, dtype=np.str_)
    color_list[(y3_pred == 1) & (y3 == 1)] = 'magenta'
    color_list[(y3_pred == 0) & (y3 == 0)] = 'cyan'
    color_list[(y3_pred == 1) & (y3 == 0)] = 'yellow'
    color_list[(y3_pred == 0) & (y3 == 1)] = 'green'
    fig = plt.figure(5)
    for i in range(4):
        for j in range(4):
            fig.add_subplot(4, 4, 4 * i + j + 1)
            if i >= j:
                plt.scatter(X3[:, i], X3[:, j], c=color_list)
            else:
                plt.scatter(X3[-1:0:-1, i], X3[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y4_pred.shape, dtype=np.str_)
    color_list[(y4_pred == 1) & (y4_exp == '1')] = 'r'
    color_list[(y4_pred == 0) & (y4_exp == '0')] = 'b'
    color_list[(y4_pred == 1) & (y4_exp == '0')] = 'y'
    color_list[(y4_pred == 0) & (y4_exp == '1')] = 'k'
    # color_list[(y == 1) & (z == 2)] = 'magenta'
    # color_list[(y == 0) & (z == 2)] = 'cyan'

    # pca = PCA(n_components=4)
    # X_pca = pca.fit_transform(X)
    fig = plt.figure(6)
    for i in range(4):
        for j in range(4):
            fig.add_subplot(4, 4, 4 * i + j + 1)
            if i >= j:
                plt.scatter(X4[color_list == 'r', i], X4[color_list == 'r', j], c='r')
                plt.scatter(X4[color_list == 'b', i], X4[color_list == 'b', j], c='b')
                plt.scatter(X4[color_list == 'y', i], X4[color_list == 'y', j], c='yellow')
                plt.scatter(X4[color_list == 'k', i], X4[color_list == 'k', j], c='k')
                plt.scatter(Xp[class_point, i], Xp[class_point, j], c='g')
                plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
                plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            else:
                # plt.scatter(Xp[-1:0:-1, i], Xp[-1:0:-1, j], c=color_list[-1:0:-1])
                plt.scatter(X4[color_list == 'k', j], X4[color_list == 'k', i], c='k')
                plt.scatter(X4[color_list == 'y', j], X4[color_list == 'y', i], c='y')
                plt.scatter(X4[color_list == 'b', j], X4[color_list == 'b', i], c='b')
                plt.scatter(X4[color_list == 'r', j], X4[color_list == 'r', i], c='r')
                plt.scatter(Xp[class_point, j], Xp[class_point, i], c='g')
                plt.xlim(np.min(X[:, j]), np.max(X[:, j]))
                plt.ylim(np.min(X[:, i]), np.max(X[:, i]))

            #plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            #plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def test_even_simpler_distribution_experiment_simulated():
    X, y, z = simulate_simple_classified(seed_num=52146, size=10000, clusters_keep=3)
    X2, y2, _ = simulate_simple_classified(seed_num=89213, size=1000, clusters_keep=1)
    X3, y3, _ = simulate_simple_classified(seed_num=78761, size=1000, clusters_keep=2)
    X4, y4, z4 = simulate_simple_classified(seed_num=11487, size=750, clusters_keep=3)
    print(X)
    print(y)
    print(sum(y))
    X = pd.DataFrame(X, columns=[str(i) for i in range(1, 2+1)])
    X2 = pd.DataFrame(X2, columns=[str(i) for i in range(1, 2+1)])
    X3 = pd.DataFrame(X3, columns=[str(i) for i in range(1, 2+1)])
    X4 = pd.DataFrame(X4, columns=[str(i) for i in range(1, 2+1)])
    # rounded linear regression performs poorly
    # a decision tree is surprisingly well suited to this problem (almost perfect)
    # SVC with rbf kernel performs ok
    # MLP performs very well when alpha is high (1e-3)
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-3,
                        hidden_layer_sizes=(10, 8, 2),
                        random_state=678993)
    # TODO: train a post-hoc explainer and compare where misclassifications occur
    #  plot the point that is being classified and circle the regions of percentages in each dimension
    #  plot the perturbed data and compare
    # clf = DecisionTreeClassifier()
    exp = BARBE(training_data=X,
                input_bounds=None,#[(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                perturbation_type='t-distribution',  # difference between uniform and normal is important here
                n_perturbations=5000,
                dev_scaling_factor=3/2,
                n_bins=10,
                verbose=False,
                input_sets_class=False)
    clf.fit(X, y)
    class_point = np.argwhere((z4 == 1) & (y4 == 1))[0][0]
    exp.explain(X4.iloc[class_point], clf)
    Xp = exp.get_perturbed_data()
    #print(Xp)
    #print(exp.get_surrogate_fidelity())
    #assert False

    y_pred = np.round(clf.predict(X))
    y2_pred = np.round(clf.predict(X2))
    y3_pred = np.round(clf.predict(X3))
    yp_pred = np.round(clf.predict(Xp))
    y4_pred = np.round(clf.predict(X4))
    y4_exp = exp.predict(X4)
    # show the quality of fit and difficulty on this data
    print("Evaluation for X")
    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print("Evaluation for Group 1")
    print(accuracy_score(y2, y2_pred))
    print(confusion_matrix(y2, y2_pred))
    print("Evaluation for Group 2")
    print(accuracy_score(y3, y3_pred))
    print(confusion_matrix(y3, y3_pred))
    print("Evaluation Compared to Perturber")
    print(accuracy_score(exp._blackbox_classification['perturbed'], exp._surrogate_classification['perturbed']))
    print(confusion_matrix(exp._blackbox_classification['perturbed'], exp._surrogate_classification['perturbed']))
    print("Evaluation Compared to Perturber on Holdout")
    print(accuracy_score(y4_pred, y4_exp.astype(int)))
    print(confusion_matrix(y4_pred, y4_exp.astype(int)))
    X = X.to_numpy()
    X2 = X2.to_numpy()
    X3 = X3.to_numpy()
    Xp = Xp.to_numpy()
    X4 = X4.to_numpy()

    # plot the results by class and cluster
    color_list = np.empty(shape=y.shape, dtype=np.str_)
    color_list[(y == 1) & (z == 1)] = 'red'
    color_list[(y == 0) & (z == 1)] = 'blue'
    color_list[(y == 1) & (z == 2)] = 'magenta'
    color_list[(y == 0) & (z == 2)] = 'cyan'
    fig = plt.figure(1)
    fig.suptitle("Actual Classes and Groups from Data")
    for i in range(2):
        for j in range(2):
            fig.add_subplot(2, 2, 2 * i + j + 1)
            if i >= j:
                plt.scatter(X[:, i], X[:, j], c=color_list)
            else:
                plt.scatter(X[-1:0:-1, i], X[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=yp_pred.shape, dtype=np.str_)
    print(exp._surrogate_classification['perturbed'])
    color_list[(yp_pred == 1) & (exp._surrogate_classification['perturbed'] == '1')] = 'r'
    color_list[(yp_pred == 0) & (exp._surrogate_classification['perturbed'] == '0')] = 'b'
    color_list[(yp_pred == 1) & (exp._surrogate_classification['perturbed'] == '0')] = 'y'
    color_list[(yp_pred == 0) & (exp._surrogate_classification['perturbed'] == '1')] = 'k'
    color_list[0] = 'g'
    #color_list[(y == 1) & (z == 2)] = 'magenta'
    #color_list[(y == 0) & (z == 2)] = 'cyan'

    #pca = PCA(n_components=4)
    #X_pca = pca.fit_transform(X)
    fig = plt.figure(2)
    fig.suptitle("Perturbed Data and Explanation Classes against Black Box")
    for i in range(2):
        for j in range(2):
            fig.add_subplot(2, 2, 2 * i + j + 1)
            if i >= j:
                plt.scatter(Xp[color_list == 'r', i], Xp[color_list == 'r', j], c='r')
                plt.scatter(Xp[color_list == 'b', i], Xp[color_list == 'b', j], c='b')
                plt.scatter(Xp[color_list == 'y', i], Xp[color_list == 'y', j], c='yellow')
                plt.scatter(Xp[color_list == 'k', i], Xp[color_list == 'k', j], c='k')
                plt.scatter(Xp[0, i], Xp[0, j], c='g')
                plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
                plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            else:
                #plt.scatter(Xp[-1:0:-1, i], Xp[-1:0:-1, j], c=color_list[-1:0:-1])
                plt.scatter(Xp[color_list == 'k', j], Xp[color_list == 'k', i], c='k')
                plt.scatter(Xp[color_list == 'y', j], Xp[color_list == 'y', i], c='y')
                plt.scatter(Xp[color_list == 'b', j], Xp[color_list == 'b', i], c='b')
                plt.scatter(Xp[color_list == 'r', j], Xp[color_list == 'r', i], c='r')
                plt.scatter(Xp[0, j], Xp[0, i], c='g')
                plt.xlim(np.min(X[:, j]), np.max(X[:, j]))
                plt.ylim(np.min(X[:, i]), np.max(X[:, i]))



            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y.shape, dtype=np.str_)
    color_list[(y_pred == 1) & (z == 1)] = 'red'
    color_list[(y_pred == 0) & (z == 1)] = 'blue'
    color_list[(y_pred == 1) & (z == 2)] = 'magenta'
    color_list[(y_pred == 0) & (z == 2)] = 'cyan'
    fig = plt.figure(3)
    fig.suptitle("Black Box Predictions of Actual Classes and Groups")
    for i in range(2):
        for j in range(2):
            fig.add_subplot(2, 2, 2 * i + j + 1)
            if i >= j:
                plt.scatter(X[:, i], X[:, j], c=color_list)

            else:
                plt.scatter(X[-1:0:-1, i], X[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y2.shape, dtype=np.str_)
    color_list[(y2_pred == 1) & (y2 == 1)] = 'red'
    color_list[(y2_pred == 0) & (y2 == 0)] = 'blue'
    color_list[(y2_pred == 1) & (y2 == 0)] = 'yellow'
    color_list[(y2_pred == 0) & (y2 == 1)] = 'green'
    fig = plt.figure(4)
    fig.suptitle("Correctness of Black Box Predictions on New Group 1 Data")
    for i in range(2):
        for j in range(2):
            fig.add_subplot(2, 2, 2 * i + j + 1)
            if i >= j:
                plt.scatter(X2[:, i], X2[:, j], c=color_list)
            else:
                plt.scatter(X2[-1:0:-1, i], X2[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y3.shape, dtype=np.str_)
    color_list[(y3_pred == 1) & (y3 == 1)] = 'magenta'
    color_list[(y3_pred == 0) & (y3 == 0)] = 'cyan'
    color_list[(y3_pred == 1) & (y3 == 0)] = 'yellow'
    color_list[(y3_pred == 0) & (y3 == 1)] = 'green'
    fig = plt.figure(5)
    fig.suptitle("Correctness of Black Box Predictions on New Group 2 Data")
    for i in range(2):
        for j in range(2):
            fig.add_subplot(2, 2, 2 * i + j + 1)
            if i >= j:
                plt.scatter(X3[:, i], X3[:, j], c=color_list)
            else:
                plt.scatter(X3[-1:0:-1, i], X3[-1:0:-1, j], c=color_list[-1:0:-1])
            plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()

    color_list = np.empty(shape=y4_pred.shape, dtype=np.str_)
    color_list[(y4_pred == 1) & (y4_exp == '1')] = 'r'
    color_list[(y4_pred == 0) & (y4_exp == '0')] = 'b'
    color_list[(y4_pred == 1) & (y4_exp == '0')] = 'y'
    color_list[(y4_pred == 0) & (y4_exp == '1')] = 'k'
    # color_list[(y == 1) & (z == 2)] = 'magenta'
    # color_list[(y == 0) & (z == 2)] = 'cyan'

    # pca = PCA(n_components=4)
    # X_pca = pca.fit_transform(X)
    fig = plt.figure(6)
    fig.suptitle("Comparison of Predictions on New Data Between Black Box and Explainer")
    for i in range(2):
        for j in range(2):
            fig.add_subplot(2, 2, 2 * i + j + 1)
            if i >= j:
                plt.scatter(X4[color_list == 'r', i], X4[color_list == 'r', j], c='r')
                plt.scatter(X4[color_list == 'b', i], X4[color_list == 'b', j], c='b')
                plt.scatter(X4[color_list == 'y', i], X4[color_list == 'y', j], c='yellow')
                plt.scatter(X4[color_list == 'k', i], X4[color_list == 'k', j], c='k')
                plt.scatter(Xp[0, i], Xp[0, j], c='g')
                plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
                plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            else:
                # plt.scatter(Xp[-1:0:-1, i], Xp[-1:0:-1, j], c=color_list[-1:0:-1])
                plt.scatter(X4[color_list == 'k', j], X4[color_list == 'k', i], c='k')
                plt.scatter(X4[color_list == 'y', j], X4[color_list == 'y', i], c='y')
                plt.scatter(X4[color_list == 'b', j], X4[color_list == 'b', i], c='b')
                plt.scatter(X4[color_list == 'r', j], X4[color_list == 'r', i], c='r')
                plt.scatter(Xp[0, j], Xp[0, i], c='g')
                plt.xlim(np.min(X[:, j]), np.max(X[:, j]))
                plt.ylim(np.min(X[:, i]), np.max(X[:, i]))

            #plt.xlim(np.min(X[:, i]), np.max(X[:, i]))
            #plt.ylim(np.min(X[:, j]), np.max(X[:, j]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def test_distribution_simulation():
    X, y, z = simulate_linear_classified(seed_num=52146, size=1000, clusters_keep=3)
    Xs, ys, _ = simulate_linear_classified(seed_num=11267, size=100, clusters_keep=3)
    X2, y2, _ = simulate_linear_classified(seed_num=89213, size=250, clusters_keep=3)
    X = pd.DataFrame(X, columns=[str(i) for i in range(1, 4 + 1)])
    X2 = pd.DataFrame(X2, columns=[str(i) for i in range(1, 4 + 1)])
    Xs = pd.DataFrame(Xs, columns=[str(i) for i in range(1, 4 + 1)])

    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-3,
                        hidden_layer_sizes=(8, 3, 3),
                        random_state=678993)
    clf.fit(X, y)

    X2 = X2.sample(frac=1)
    distribution_experiment(Xs, ys, X2.iloc[0:20], X2.iloc[20:],
                            n_perturbations=1000,
                            pre_trained_model=clf, local_test_end=20,
                            data_name="simulation")


def test_distribution_experiment_loan():
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/train_loan_raw.csv")
    data = data.drop('Loan_ID', axis=1)
    print(list(data))
    data = data.dropna()
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    data = data.dropna()
    data = data.sample(frac=1, random_state=117283)
    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Loan_Status']:
            data[cat] = data[cat].astype(float)

    y = data['Loan_Status']
    data = data.drop(['Loan_Status'], axis=1)

    preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    model = Pipeline([('pre', preprocess),
                      ('clf', MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257))])
                      #('clf', RandomForestClassifier(n_estimators=5,
                      #                               max_depth=4,
                      #                               min_samples_split=10,
                      #                               min_samples_leaf=3,
                      #                               bootstrap=True,
                      #                               random_state=301257))])

    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_training_stop = int(loan_training.shape[0] // 3)

    for feature in list(loan_training):
        unique_values = np.unique(loan_training.iloc[0:loan_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
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
    model.fit(loan_training, loan_training_label)
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
                distribution_experiment(loan_training,
                                        loan_training_label,
                                        loan_perturb,
                                        loan_test,
                                        pre_trained_model=model,
                                        # lime_version=LimeNewPert,
                                        local_test_end=50,
                                        data_name="barbe_neural_loan_acceptance",
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


def test_distribution_experiment_libras():
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
if __name__ == '__name__':
    simple_distribution_experiment_simulated()


# TODO: add a timeseries dataset
# TODO: test on more datasets with >2 labels
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
