# IAIN make this file get and store results for each of the fidelity values we will check
#  against each other for each distribution type + lime1's distribution
import dill

# experiments to see which of Lime, Uniform, Standard-Normal, Multi-Normal, Clustered-Normal, t-Distribution, Chauchy
#  perform the best in terms of fidelity values

# pert_fidelity = barbe[i].fidelity() -> training accuracy
# train_fidelity = barbe[i].fidelity(training_data, bbmodel) -> single blind / validation accuracy
# test_fidelity = barbe[i].fidelity(bbmodel_data, bbmodel) -> double blind / testing accuracy

from barbe.utils.lime_interface import LimeNewPert, VAELimeNewPert, SLimeNewPert
from barbe.utils.dummy_interfaces import DummyExplainer
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

def get_explanation_features_lore(lore_exp):
    return None


from itertools import combinations
import numpy as np


def calculate_stability_baseline(nfeatures, npicks, nreps, nruns):

    stab_softn = []
    stab_soft5 = []
    stab_hard = []
    possible_feature_list = np.array([i for i in range(nfeatures)])
    for i in range(nruns):
        temp_choices = [np.random.choice(possible_feature_list, size=npicks, replace=False)
                                                    for _ in range(nreps)]
        temp_choices = [[(temp_choices[j][i], nfeatures-i) for i in range(len(temp_choices[j]))] for j in range(len(temp_choices))]
        #print(temp_choices)
        stab_softn.append(calculate_loose_stability(temp_choices, nfeatures))
        stab_soft5.append(calculate_loose_stability(temp_choices, 10))
        stab_hard.append(calculate_stability(temp_choices))

    stab_final_hard = [[i, 0] for i in range(5)]
    #print(stab_hard)
    for j in range(5):
        for i in range(nruns):
            stab_final_hard[j][1] += stab_hard[i][j] / nruns

    return np.mean(stab_soft5), np.mean(stab_softn), stab_final_hard


def test_output_baselines():
    soft5, softn, ranking = calculate_stability_baseline(20, 20, 10, 50)
    #for i in range(30, 15-1, -1):
    #    soft, hard = calculate_stability_baseline(i, 15, 10, 50)
    #    print("Selecting From: ", i, " features")
    print("Soft 5: ", soft5)
    print("Soft N: ", softn)
    print("Ranking: ", ranking)
     #   print("Sqrt Hard: ", hard**(1/2))


def calculate_variance_stability(nrun_features, all_feature_names):
    final_avg = 0
    final_counts = 0
    variance_dict = {}
    average_dict = {}
    for feature in all_feature_names:
        average_dict[feature] = None
        variance_dict[feature] = None

    total_runs = len(nrun_features)

    for i in range(total_runs):
        nrun_i_top = nrun_features[i][0:(len(all_feature_names)//2)]
        for feat, impt in nrun_i_top:
            if average_dict[feat] is None:
                average_dict[feat] = 0
            average_dict[feat] += impt

    for i in range(total_runs):
        nrun_i_top = nrun_features[i][0:(len(all_feature_names) // 2)]
        for feat, impt in nrun_i_top:
            if variance_dict[feat] is None:
                variance_dict[feat] = 0
            variance_dict[feat] += (((average_dict[feat]/total_runs) - impt)**2)/(total_runs-1)

    for _, value in variance_dict.items():
        final_counts += 1
        final_avg += value
    return final_avg/final_counts


def calculate_stability(nrun_features):
    #print(nrun_features)
    top_ranked_stability = [0 for _ in range(5)]
    for i in range(5):
        total_adds = 0
        for j in range(len(nrun_features)):
            for k in range(j+1, len(nrun_features)):
                if len(nrun_features[j]) > 0 and len(nrun_features[k]) > 0:
                    total_adds += 1
                    #print(nrun_features[j][i][0])
                    #print(nrun_features[k][i][0])
                    #print(set([nrun_features[j][i][0]]).intersection(set([nrun_features[k][i][0]])))
                    #print(set([nrun_features[j][i][0]]).union(set([nrun_features[k][i][0]])))
                    top_ranked_stability[i] += len(set([nrun_features[j][i][0]]).intersection(set([nrun_features[k][i][0]]))) / len(set([nrun_features[j][i][0]]).union(set([nrun_features[k][i][0]])))
        if total_adds != 0:
            top_ranked_stability[i] /= total_adds
        else:
            return None

    # return the full list we will average the top rank stabilities at the end (for table)
    return top_ranked_stability


def calculate_loose_stability(nrun_features, total_dataset_features):
    #nrun_features = [feat for feat, _ in nrun_features]
    #nrun_features = nrun_features[0:(total_dataset_features // 2)]
    jaccard_sum = 0
    n_counts = 0
    for i in range(len(nrun_features)):
        row1_features = [feat for feat, _ in nrun_features[i]][0:(total_dataset_features // 2)]
        row1 = set(row1_features)
        if len(row1) > 0:
            for j in range(i+1, len(nrun_features)):
                row2_features = [feat for feat, _ in nrun_features[j]][0:(total_dataset_features // 2)]
                row2 = set(row2_features)
                if len(row2) > 0:
                    jaccard_sum += (len(row1.intersection(row2)) / len(row1.union(row2)))
                    n_counts += 1
    if n_counts == 0:
        return None
    return jaccard_sum / n_counts


def get_elbow(feature_importance, diff_tol=0.0001):
    # look through all the data and once the differences start getting lesser then stop
    #prev_diff = feature_importance[0][1] - feature_importance[1][1]
    i = 4
    while i < len(feature_importance)-2 and not (feature_importance[i+1][1] == 0) and not (1 - diff_tol < feature_importance[i][1]/feature_importance[i+1][1] < 1 + diff_tol):
        i += 1
    while i < len(feature_importance)-2 and not (feature_importance[i+1][1] == 0) and (1 - diff_tol < feature_importance[i][1] / feature_importance[i + 1][1] < 1 + diff_tol) :
        i += 1
    return i + 1


def get_lime_elbow(feature_importance, diff_tol=0.1):
    # look through all the data and once the differences start getting lesser then stop
    #prev_diff = feature_importance[0][1] - feature_importance[1][1]
    i = 4
    while i < len(feature_importance)-2 and not (feature_importance[i+1][1] == 0) and not (1 - diff_tol < feature_importance[i][1]/feature_importance[i+1][1] < 1 + diff_tol):
        i += 1
    return i + 1


def baseline_distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                            discrete_features=None,
                            local_test_end=20, data_name="data", n_perturbations=100, n_bins=5, dev_scaling=10, use_class_balance=False):
    # TODO: make this set up folds to run the experiments over and then be more generic
    random_seeds = [117893, 7858801, 3256767, 98451, 787,
                    9631, 4482, 999999991, 30592, 123411]
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
    # barbe_dist = ['standard-normal', 'normal']
    barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end * 10)]
    fidelity_single_blind = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind = [[] for _ in range(local_test_end * 10)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end * 10)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end * 10)]

    fidelity_single_diff_n = [[] for _ in range(local_test_end * 10)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end * 10)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end * 10)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end * 10)]

    hit_rate = [[] for _ in range(local_test_end * 10)]
    avg_features = [[] for _ in range(local_test_end * 10)]
    stability = [[] for _ in range(local_test_end)]
    loose_stability = [[] for _ in range(local_test_end)]
    loose_stability5 = [[] for _ in range(local_test_end)]

    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i].copy()
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:
            top_features = [[] for _ in range(10)]
            for j in range(10):
                iris_wb_training = iris_training.sample(frac=1/3, random_state=random_seeds[j])
                try:
                    explainer = DummyExplainer(training_data=iris_wb_training)

                    pert_row = iris_test.iloc[i]
                    input_row = pd.DataFrame(columns=list(iris_training), index=[0])
                    #print(pert_row)
                    #print(list(iris_training))
                    input_row.iloc[0] = pert_row.to_numpy().reshape((1, -1))
                    splanation = explainer.explain(pert_row.copy(), bbmodel)
                    #print(explainer.perturber.get_balance())
                    #print(explainer.perturber.get_number_iterations())
                    #assert False
                    top_features[j] = list()

                    temp_pert = 0
                    temp_single_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_training.drop('target', axis=1,
                                                                                                        errors='ignore'),
                                                                     weights=None,
                                                                     original_data=pert_row)
                    temp_double_f = explainer.perturber.get_number_iterations()
                    fidelity_pert[i + j * local_test_end].append(temp_pert)
                    fidelity_single_blind[i + j * local_test_end].append(temp_single_f)
                    fidelity_double_blind[i + j * local_test_end].append(temp_double_f)
                    temp_single = explainer.perturber.get_balance()['Yes']
                    temp_double = explainer.perturber.get_balance()['No']
                    fidelity_single_blind_e[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_e[i + j * local_test_end].append(temp_double)

                    fidelity_single_diff_e[i + j * local_test_end].append(temp_single - temp_single_f)
                    fidelity_double_diff_e[i + j * local_test_end].append(temp_double - temp_double_f)
                    temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_training,
                                                                   weights='nearest',
                                                                   original_data=pert_row)
                    temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_wb_test,
                                                                   weights='nearest',
                                                                   original_data=pert_row)
                    fidelity_single_blind_n[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_n[i + j * local_test_end].append(temp_double)

                    fidelity_single_diff_n[i + j * local_test_end].append(temp_single - temp_single_f)
                    fidelity_double_diff_n[i + j * local_test_end].append(temp_double - temp_double_f)
                    hit_rate[i + j * local_test_end].append(1 if explainer._predict_class == 'Y' else 0)
                    #print(splanation)

                    avg_features[i + j*local_test_end].append(0)
                    #del explainer
                    #del splanation
                    #if len(top_features[j]) == 0:
                    #    print(top_features)
                    #    assert False

                except:
                    print(traceback.format_exc())
                    assert False
                    print("Failed")
                    temp_pert = None
                    temp_single = None
                    temp_double = None
                    fidelity_pert[i+ j * local_test_end].append(temp_pert)
                    fidelity_single_blind[i+ j * local_test_end].append(temp_single)
                    fidelity_double_blind[i+ j * local_test_end].append(temp_double)
                    fidelity_single_blind_e[i+ j * local_test_end].append(temp_single)
                    fidelity_double_blind_e[i+ j * local_test_end].append(temp_double)
                    fidelity_single_blind_n[i+ j * local_test_end].append(temp_single)
                    fidelity_double_blind_n[i+ j * local_test_end].append(temp_double)
                    # NEW
                    fidelity_single_diff_n[i+ j * local_test_end].append(None)
                    fidelity_single_diff_e[i+ j * local_test_end].append(None)
                    fidelity_double_diff_n[i+ j * local_test_end].append(None)
                    fidelity_double_diff_e[i+ j * local_test_end].append(None)
                    hit_rate[i+ j * local_test_end].append(0)
                    avg_features[i + j * local_test_end].append(None)

                #print(i, " - ", distribution)
                #print(temp_pert)
                #print(temp_single)
                #print(temp_double)
            stability[i].append([0, 0, 0, 0, 0])
            loose_stability[i].append(0)
            loose_stability5[i].append(0)
            #print(calculate_loose_stability(top_features))
            #assert False
            #for feats in top_features:
            #    print(feats)
            #print(calculate_stability(top_features))
            #assert False
            #if calculate_stability(top_features) == 0:
            #    print(top_features)
            #    assert False

    # NEW
    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate",
                       "Avg. Features",
                       "Stability 1", "Stab. 1 Var.", "Stability 2", "Stab. 2 Var.",
                       "Stability 3", "Stab. 3 Var.", "Stability 4", "Stab. 4 Var.",
                       "Stability 5", "Stab. 5 Var.",
                       "Loose Stab.", "Loose Stab. Var.",
                       "Loose Stab. 5", "Loose Stab. 5 Var."]]
    print(fidelity_pert)
    print(fidelity_single_blind)
    print(fidelity_double_blind)
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
                for k in range(10):
                    print("WHY ", fidelity_pert, i, k, j)
                    if fidelity_pert[i + k * local_test_end][j] is not None:
                        mean_count += 1
                        temp_pert_acc += fidelity_pert[i + k * local_test_end][j]
                        temp_single_acc += fidelity_single_blind[i + k * local_test_end][j]
                        temp_double_acc += fidelity_double_blind[i + k * local_test_end][j]
            temp_pert_std = 0
            temp_single_std = 0
            temp_double_std = 0
            for i in range(local_test_end):
                for k in range(10):
                    if fidelity_pert[i + k * local_test_end][j] is not None:
                        temp_pert_std += (fidelity_pert[i + k * local_test_end][j] - temp_pert_acc / mean_count) ** 2
                        temp_single_std += (fidelity_single_blind[i + k * local_test_end][
                                                j] - temp_single_acc / mean_count) ** 2
                        temp_double_std += (fidelity_double_blind[i + k * local_test_end][
                                                j] - temp_double_acc / mean_count) ** 2
            if mean_count == 1:
                temp_pert_std = 0
                temp_single_std = 0
                temp_double_std = 0
            else:
                temp_pert_std = (temp_pert_std / (mean_count - 1))
                temp_single_std = (temp_single_std / (mean_count - 1))
                temp_double_std = (temp_double_std / (mean_count - 1))
            # print(barbe_dist[j])
            # add standard deviations as info
            if (len(averages_print) - 1) / 3 <= j:
                averages_print.append([barbe_dist[j], "Perturbed"])
                averages_print.append([barbe_dist[j], "Single Blind"])
                averages_print.append([barbe_dist[j], "Double Blind"])
            if mean_count != 0:
                averages_print[(j * 3) + 1].append(temp_pert_acc / mean_count)
                averages_print[(j * 3) + 1].append(temp_pert_std)
                averages_print[(j * 3) + 2].append(temp_single_acc / mean_count)
                averages_print[(j * 3) + 2].append(temp_single_std)
                averages_print[(j * 3) + 3].append(temp_double_acc / mean_count)
                averages_print[(j * 3) + 3].append(temp_double_std)
            else:
                averages_print[(j * 3) + 1].append(0)
                averages_print[(j * 3) + 1].append(0)
                averages_print[(j * 3) + 2].append(0)
                averages_print[(j * 3) + 2].append(0)
                averages_print[(j * 3) + 3].append(0)
                averages_print[(j * 3) + 3].append(0)
    for j in range(len(barbe_dist)):
        average_hits = 0
        for i in range(local_test_end):
            for k in range(10):
                average_hits += hit_rate[i + k * local_test_end][j]
        average_hits /= local_test_end * 10
        averages_print[(j * 3) + 1].append(average_hits)
        averages_print[(j * 3) + 2].append(0)
        averages_print[(j * 3) + 3].append(0)
    for j in range(len(barbe_dist)):
        average_features = 0
        avg_count = 0
        for i in range(local_test_end):
            for k in range(10):
                if avg_features[i + k * local_test_end][j] is not None:
                    avg_count += 1
                    average_features += avg_features[i + k * local_test_end][j]
        if avg_count != 0:
            average_features /= avg_count
        averages_print[(j * 3) + 1].append(average_features)
        averages_print[(j * 3) + 2].append(-1)
        averages_print[(j * 3) + 3].append(-1)

    for k in range(5):
        for j in range(len(barbe_dist)):
            average_stab = 0
            for i in range(local_test_end):
                average_stab += stability[i][j][k]
            average_stab /= local_test_end
            averages_print[(j * 3) + 1].append(average_stab)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)
            stab_var = 0
            for i in range(local_test_end):
                stab_var += (stability[i][j][k] - average_stab) ** 2
            stab_var /= local_test_end - 1
            averages_print[(j * 3) + 1].append(stab_var)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)

    for j in range(len(barbe_dist)):
        average_stab = 0
        for i in range(local_test_end):
            average_stab += loose_stability[i][j]
        average_stab /= local_test_end
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            stab_var += (loose_stability[i][j] - average_stab) ** 2
        stab_var = (stab_var / (local_test_end - 1)) if local_test_end != 1 else 0
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
    for j in range(len(barbe_dist)):
        average_stab = 0
        for i in range(local_test_end):
            average_stab += loose_stability5[i][j]
        average_stab /= local_test_end
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            stab_var += (loose_stability5[i][j] - average_stab) ** 2
        stab_var = (stab_var / (local_test_end - 1)) if local_test_end != 1 else 0
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
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

    pd.DataFrame(averages_print).to_csv("Results/baseline_"+"_".join([data_name,
                                                             "nruns"+str(local_test_end),
                                                             "nbins"+str(n_bins),
                                                             "nperturb"+str(n_perturbations),
                                                             "devscalin"+str(dev_scaling)])+"_results.csv")
    #print([(np.nanmin(iris_numpy[:,0]), np.nanmax(iris_numpy[:,0])),
    #                                            (np.nanmin(iris_numpy[:,1]), np.nanmax(iris_numpy[:,1])),
    #                                            (np.nanmin(iris_numpy[:,2]), np.nanmax(iris_numpy[:,2])),
    #                                            (np.nanmin(iris_numpy[:,3]), np.nanmax(iris_numpy[:,3])),])



def lore_distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, dev_scaling=1.0,
                                 local_test_end=20, data_name="data", n_perturbations=100,
                                 use_barbe_perturbations=False,
                                 random_seeds=None):
    # TODO: make this set up folds to run the experiments over and then be more generic

    # LIME requires more steps preprocess the data here
    #cat_encoder = CategoricalEncoder(ordinal_encoding=False)
    #iris_training = cat_encoder.fit_transform(training_data=iris_training)
    #iris_perturb = cat_encoder.transform(iris_perturb)
    #iris_test = cat_encoder.transform(iris_test)
    #print(cat_encoder._encoder_key)
    if random_seeds is None:
        random_seeds = [117893, 7858801, 3256767, 98451, 787,
                        9631, 4482, 999999991, 30592, 123411]

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
    # part_training = int(iris_training.shape[0] // 3)  # specifically for the LOAN


    # barbe_dist = ['standard-normal', 'normal', 'uniform', 'cauchy', 't-distribution'] if use_barbe_perturbations else ['pert-lime']
    barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end*10)]
    fidelity_single_blind = [[] for _ in range(local_test_end*10)]
    fidelity_double_blind = [[] for _ in range(local_test_end*10)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end*10)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end*10)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end*10)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end*10)]

    fidelity_single_diff_n = [[] for _ in range(local_test_end*10)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end*10)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end*10)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end*10)]

    hit_rate = [[] for _ in range(local_test_end*10)]
    avg_features = [[] for _ in range(local_test_end * 10)]
    stability = [[] for _ in range(local_test_end)]
    loose_stability = [[] for _ in range(local_test_end)]
    loose_stability5 = [[] for _ in range(local_test_end)]

    iris_category_features = list(np.where(np.isin(np.array(list(iris_training)), discrete_features))[0])
    print("Processed Categories: ", iris_category_features)

    iris_add_to_training = iris_training.iloc[0:1]
    iris_test['target'] = bbmodel.predict(iris_test.drop('target', axis=1, errors='ignore'))
    iris_training['target'] = bbmodel.predict(iris_training.drop('target', axis=1, errors='ignore'))
    #print("Uniques: ", np.unique(iris_training['target']))
    #print("Uniques: ", np.unique(iris_test['target']))
    #assert False
    for c in np.unique(iris_training['target']):
        class_position = np.where(iris_training['target'] == c)[0]
        print("CLASS POSITION: ", class_position)
        print("CLASS VALUE: ", c)
        iris_add_to_training = iris_add_to_training.append(iris_training.iloc[class_position[0:2]], ignore_index=True)
    iris_training.drop('target', inplace=True, axis=1)
    iris_test.drop('target', inplace=True, axis=1, errors='ignore')

    #iris_perturb['target'] = bbmodel.predict(iris_perturb)
    for i in range(local_test_end):  # use when removing LIME
        # i = 31  # loan 9 is female
        #pert_row = iris_perturb.drop('target', inplace=False, axis=1).iloc[i]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        iris_wb_test = iris_wb_test.drop('target', inplace=False, axis=1, errors='ignore')
        top_features = [[] for _ in range(10)]
        for j in range(10):
            print("Indiv: ", i, ", Rep: ", j)
            iris_wb_training = iris_training.sample(frac=1/3, random_state=random_seeds[j])
            iris_wb_training = iris_wb_training.append(iris_add_to_training, ignore_index=True)
            try:
                #assert False
                iris_numpy = iris_training.to_numpy()
                #print(np.unique(bbmodel.predict(iris_training)))
                #print(np.unique(iris_training_label))
                #print(np.unique(bbmodel.predict(iris_test)))
                #assert False
                explainer = LoreExplainer(iris_wb_training)
                print("Sample to Perturb: ", iris_test.iloc[i])
                splanation, info = explainer.explain(input_data=iris_test,
                                                     input_index=i,
                                                     df=iris_wb_training,
                                                     df_labels=list(bbmodel.predict(iris_training)),
                                                     blackbox=bbmodel,
                                                     discrete_use_probabilities=True)  # IAIN see if this works

                print("IAIN RULES: ", splanation)
                #if j > 8:
                #    assert False
                #print(info)
                #print(info['dt'].edges())
                #print(info['dt'].nodes())
                #print(splanation)
                #assert False
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
                fidelity_pert[i+j*local_test_end].append(temp_pert)
                fidelity_single_blind[i+j*local_test_end].append(temp_single_f)
                fidelity_double_blind[i+j*local_test_end].append(temp_double_f)
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training.drop('target', axis=1, errors='ignore'),
                                                               weights='euclidean',
                                                               original_data=pert_row)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test.drop('target', axis=1, errors='ignore'),
                                                               weights='euclidean',
                                                               original_data=pert_row)
                fidelity_single_blind_e[i+j*local_test_end].append(temp_single)
                fidelity_double_blind_e[i+j*local_test_end].append(temp_double)

                fidelity_single_diff_e[i+j*local_test_end].append(temp_single - temp_single_f)
                fidelity_double_diff_e[i+j*local_test_end].append(temp_double - temp_double_f)
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_training,
                                                               weights='nearest',
                                                               original_data=pert_row)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                               comparison_data=iris_wb_test,
                                                               weights='nearest',
                                                               original_data=pert_row)
                fidelity_single_blind_n[i+j*local_test_end].append(temp_single)
                fidelity_double_blind_n[i+j*local_test_end].append(temp_double)

                fidelity_single_diff_n[i+j*local_test_end].append(temp_single - temp_single_f)
                fidelity_double_diff_n[i+j*local_test_end].append(temp_double - temp_double_f)
                hit_rate[i+j*local_test_end].append(1)
                #print(splanation)
                #splanation = sorted(splanation[0][1].items(), key=lambda x: x[1], reverse=True)
                #print(splanation)
                #assert False
                #splanation = [feat for feat, _ in splanation]
                #if len(splanation) > 5:
                #    splanation = splanation[0:5]
                feature_dict = {}
                for item in list(iris_test):
                    feature_dict[item] = 0
                for key in splanation[0][1].keys():
                    feature_dict[key] += 1
                for crule in splanation[1]:
                    for key in crule.keys():
                        feature_dict[key] += 1

                temp_order = [(key, value) for key, value in feature_dict.items()]
                temp_order = sorted(temp_order, key=lambda x: abs(x[1]), reverse=True)
                start_zero = None
                for k in range(len(temp_order)):
                    if temp_order[k][1] == 0 or start_zero is not None:
                        if start_zero is None:
                            start_zero = k
                        swap = random.randint(start_zero, len(temp_order)-1)
                        temp_hold = temp_order[swap]
                        temp_order[swap] = temp_order[k]
                        temp_order[k] = temp_hold
                avg_features[i + j * local_test_end].append(start_zero)

                new_splanation = {}
                for feat, val in temp_order:
                    if "=" in feat:
                        new_feature = feat.split("=")[0]
                        if new_feature not in new_splanation.keys():
                            new_splanation[new_feature] = 0
                        new_splanation[new_feature] += val
                    else:
                        new_splanation[feat] = val
                splanation = [(feat, val) for feat, val in new_splanation.items()]

                top_features[j] = splanation
                #print(top_features[j])
                #assert False
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
                # assert False
                # YOU CANNOT RANDOMLY DECIDE TO THROW OUT A NEW ERROR AND EXPECT ME TO STILL GIVE
                #  THE BENIFIT OF THE DOUBT, THIS WILL NOW BE CONSIDERED A FAILURE AND WILL BE
                #  TREATED AS SUCH. THIS SHIT IS RIDICULOUS, TAKING 8 GAWDAMN HOURS ASS.
                #print(traceback.format_exc())
                #if (str(explainer.predict([input_row.to_dict('records')[0]])[0]) ==
                #        str(bbmodel.predict(input_row)[0])):
                #    print(traceback.format_exc())
                #    assert False
                #assert False
                temp_pert = None
                temp_single = None
                temp_double = None
                fidelity_pert[i+j*local_test_end].append(temp_pert)
                fidelity_single_blind[i+j*local_test_end].append(temp_single)
                fidelity_double_blind[i+j*local_test_end].append(temp_double)
                fidelity_single_blind_e[i+j*local_test_end].append(temp_single)
                fidelity_double_blind_e[i+j*local_test_end].append(temp_double)
                fidelity_single_blind_n[i+j*local_test_end].append(temp_single)
                fidelity_double_blind_n[i+j*local_test_end].append(temp_double)
                fidelity_double_diff_e[i+j*local_test_end].append(None)
                fidelity_single_diff_e[i+j*local_test_end].append(None)
                fidelity_double_diff_n[i+j*local_test_end].append(None)
                fidelity_single_diff_n[i+j*local_test_end].append(None)

                hit_rate[i+j*local_test_end].append(0)
                avg_features[i + j * local_test_end].append(None)

        stability[i].append(calculate_stability(top_features))
        loose_stability[i].append(calculate_loose_stability(top_features, len(list(iris_test))))
        loose_stability5[i].append(calculate_loose_stability(top_features, 10))
        #print(stability)
        #assert False

    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate", "Avg. Features",
                       "Stability 1", "Stab. 1 Var.", "Stability 2", "Stab. 2 Var.",
                       "Stability 3", "Stab. 3 Var.", "Stability 4", "Stab. 4 Var.",
                       "Stability 5", "Stab. 5 Var.",
                       "Loose Stab.", "Loose Stab. Var",
                       "Loose Stab. 5", "Loose Stab. 5 Var"]]

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
            for i in range(local_test_end*1):
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

                # add standard deviations as info
            if (len(averages_print) - 1) / 3 <= j:
                averages_print.append([barbe_dist[j], "Perturbed"])
                averages_print.append([barbe_dist[j], "Single Blind"])
                averages_print.append([barbe_dist[j], "Double Blind"])
            if mean_count != 0:
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

    for j in range(len(barbe_dist)):
        average_features = 0
        avg_count = 0
        for i in range(local_test_end):
            for k in range(10):
                if avg_features[i + k * local_test_end][j] is not None:
                    avg_count += 1
                    average_features += avg_features[i + k * local_test_end][j]
        if avg_count != 0:
            average_features /= avg_count
        averages_print[(j * 3) + 1].append(average_features)
        averages_print[(j * 3) + 2].append(-1)
        averages_print[(j * 3) + 3].append(-1)

    for k in range(5):
        for j in range(len(barbe_dist)):
            average_stab = 0
            stab_count = 0
            for i in range(local_test_end):
                if stability[i][j] is not None:
                    average_stab += stability[i][j][k]
                    stab_count += 1
            if stab_count != 0:
                average_stab /= stab_count
            averages_print[(j * 3) + 1].append(average_stab)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)
            stab_var = 0
            for i in range(local_test_end):
                if stability[i][j] is not None:
                    stab_var += (stability[i][j][k] - average_stab) ** 2
            if stab_count > 1:
                stab_var /= stab_count - 1
            averages_print[(j * 3) + 1].append(stab_var)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)

    for j in range(len(barbe_dist)):
        average_stab = 0
        stab_count = 0
        for i in range(local_test_end):
            if loose_stability[i][j] is not None:
                average_stab += loose_stability[i][j]
                stab_count += 1
        if stab_count != 0:
            average_stab /= stab_count
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            if loose_stability[i][j] is not None:
                stab_var += (loose_stability[i][j] - average_stab) ** 2
        if stab_count > 1:
            stab_var = (stab_var / (stab_count - 1))
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)

    for j in range(len(barbe_dist)):
        average_stab = 0
        stab_count = 0
        for i in range(local_test_end):
            if loose_stability5[i][j] is not None:
                average_stab += loose_stability5[i][j]
                stab_count += 1
        if stab_count != 0:
            average_stab /= stab_count
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            if loose_stability5[i][j] is not None:
                stab_var += (loose_stability5[i][j] - average_stab) ** 2
        if stab_count > 1:
            stab_var = (stab_var / (stab_count - 1))
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)


    pd.DataFrame(averages_print).to_csv("Results/lore_" + "_".join([data_name,
                                                               "nruns" + str(local_test_end),
                                                               "nperturb" + str(n_perturbations),
                                                               "barbe" + str(use_barbe_perturbations),
                                                               "devscalin" + str(dev_scaling)]) + "_results.csv")



def lime_distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                                 discrete_features=None, dev_scaling=1.0,
                                 local_test_end=20, data_name="data", n_perturbations=100,
                                 lime_version=LimeNewPert, lime_discretizer='decile',
                                 use_barbe_perturbations=False,
                                 use_slime=False):
    # TODO: make this set up folds to run the experiments over and then be more generic
    random_seeds = [117893, 7858801, 3256767, 98451, 787,
                    9631, 4482, 999999991, 30592, 123411]
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

    # barbe_dist = ['standard-normal', 'normal', 'uniform', 'cauchy', 't-distribution'] if use_barbe_perturbations else ['pert-lime']
    barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end * 10)]
    fidelity_single_blind = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind = [[] for _ in range(local_test_end * 10)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end * 10)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end * 10)]

    fidelity_single_diff_n = [[] for _ in range(local_test_end * 10)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end * 10)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end * 10)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end * 10)]

    hit_rate = [[] for _ in range(local_test_end * 10)]
    avg_features = [[] for _ in range(local_test_end * 10)]
    stability = [[] for _ in range(local_test_end)]
    loose_stability = [[] for _ in range(local_test_end)]
    loose_stability5 = [[] for _ in range(local_test_end)]

    iris_category_features = list(np.where(np.isin(np.array(list(iris_training)), discrete_features))[0])
    print("Processed Categories: ", iris_category_features)

    for i in range(local_test_end):  # use when removing LIME
        pert_row = iris_perturb.iloc[i]
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:
            top_features = [[] for _ in range(10)]
            for j in range(10):
                print("Subj: ", i, ", Rep: ", j)
                iris_wb_training = iris_training.sample(frac=1 / 3, random_state=random_seeds[j])
                try:
                    iris_numpy = iris_training.to_numpy()
                    #print(np.unique(bbmodel.predict(iris_training)))
                    #print(np.unique(iris_training_label))
                    #print(np.unique(bbmodel.predict(iris_test)))
                    #assert False
                    if not use_slime:
                    #    print(iris_wb_training)
                    #    assert False
                        explainer = lime_version(training_data=iris_wb_training,
                                                 training_labels=iris_training_label,
                                                 feature_names=list(iris_training),
                                                 discretizer=lime_discretizer,
                                                 discretize_continuous=False,
                                                 sample_around_instance=True)
                                                 #categorical_features=iris_category_values,
                                                 #categorical_names=iris_category_features)
                        splanation = explainer.explain_instance(data_row=pert_row,
                                                       predict_fn=lambda x: (np.eye(2, 2)[np.argmax(bbmodel.predict_proba(x), axis=1)]),
                                                       labels=(0, 1),#(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ),
                                                       num_features=iris_wb_training.shape[1],
                                                       num_samples=n_perturbations,
                                                       barbe_mode=use_barbe_perturbations,
                                                       barbe_pert_model=distribution,
                                                       barbe_dev_scaling=dev_scaling)  # IAIN see if this works
                    else:
                        explainer = SLimeNewPert(training_data=iris_wb_training,
                                                 training_labels=iris_training_label,
                                                 feature_names=list(iris_training),
                                                 discretizer=lime_discretizer,
                                                 feature_selection='lasso_path',
                                                 discretize_continuous=False,
                                                 sample_around_instance=True)
                        # categorical_features=iris_category_values,
                        # categorical_names=iris_category_features)
                        splanation = explainer.slime(data_row=pert_row,
                                                                predict_fn=lambda x: (np.eye(2, 2)[
                                                                    np.argmax(bbmodel.predict_proba(x), axis=1)]),
                                                                labels=(0, 1),# (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, ),
                                                                num_features=iris_wb_training.shape[1],
                                                                num_samples=1000,
                                                                n_max=10000,
                                                                tol=1e-3)

                    input_row = pd.DataFrame(columns=list(iris_training), index=[0])
                    input_row.iloc[0] = pert_row.to_numpy().reshape((1, -1))
                    # .to_numpy().reshape(1, -1)
                    if (str(explainer.predict(pert_row)[0]) !=
                            str(bbmodel.predict(input_row)[0])):
                        print(explainer.predict(pert_row), bbmodel.predict(input_row))
                        assert False

                    temp_pert = explainer.get_surrogate_fidelity(comparison_model=bbmodel)
                    temp_single_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_training.drop('target', axis=1,
                                                                                                        errors='ignore'),
                                                                     weights=None,
                                                                     original_data=pert_row)
                    temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_wb_test.drop('target', axis=1,
                                                                                                       errors='ignore'),
                                                                     weights=None,
                                                                     original_data=pert_row)
                    fidelity_pert[i + j * local_test_end].append(temp_pert)
                    fidelity_single_blind[i + j * local_test_end].append(temp_single_f)
                    fidelity_double_blind[i + j * local_test_end].append(temp_double_f)
                    temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_training.drop('target', axis=1,
                                                                                                      errors='ignore'),
                                                                   weights='euclidean',
                                                                   original_data=pert_row)
                    temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_wb_test.drop('target', axis=1,
                                                                                                     errors='ignore'),
                                                                   weights='euclidean',
                                                                   original_data=pert_row)
                    fidelity_single_blind_e[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_e[i + j * local_test_end].append(temp_double)

                    fidelity_single_diff_e[i + j * local_test_end].append(temp_single - temp_single_f)
                    fidelity_double_diff_e[i + j * local_test_end].append(temp_double - temp_double_f)
                    temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_training,
                                                                   weights='nearest',
                                                                   original_data=pert_row)
                    temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_wb_test,
                                                                   weights='nearest',
                                                                   original_data=pert_row)
                    fidelity_single_blind_n[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_n[i + j * local_test_end].append(temp_double)

                    fidelity_single_diff_n[i + j * local_test_end].append(temp_single - temp_single_f)
                    fidelity_double_diff_n[i + j * local_test_end].append(temp_double - temp_double_f)
                    hit_rate[i + j * local_test_end].append(1)
                    #print(splanation.as_list())
                    # something here?
                    splanation = splanation.as_list()
                    print(splanation)
                    print(explainer.perturbed_data)
                    new_splanation = {}
                    for feat, val in splanation:
                        if "=" in feat:
                            new_feature = feat.split("=")[0]
                            if new_feature not in new_splanation.keys():
                                new_splanation[new_feature] = 0
                            new_splanation[new_feature] += val
                        else:
                            new_splanation[feat] = val
                    splanation = [(feat, val) for feat, val in new_splanation.items()]
                    feature_list = []
                    to_add_features = []
                    for feat_p in list(iris_wb_training):
                        found = False
                        for feat_i, imp in splanation:
                            # print(feat_i, feat_p)
                            if not found and feat_i == feat_p:
                                # print("Compared and added: ", imp)
                                feature_list.append(imp)
                                found = True
                        if not found:
                            feature_list.append(0)
                            to_add_features.append((feat_p, 0))
                    # assert False

                    # importance_print.append(["Old Approach", "BARBE", distribution, inflection_point+1, 0,
                    #                         0, 0, 0, 0, 0, 0] + feature_list)
                    random.shuffle(to_add_features)
                    splanation = list(splanation.copy() + to_add_features)

                    elbow = get_lime_elbow(splanation)
                    #splanation = [feature for feature, _ in splanation.as_list()]
                    #splanation = splanation[0:elbow]
                    avg_features[i + j * local_test_end].append(elbow+1)
                    #if len(splanation) > 10:
                    #    splanation = splanation[0:10]
                    #print(splanation)
                    #assert False
                    top_features[j] = splanation
                except:
                    print(traceback.format_exc())
                    #if (str(explainer.predict(pert_row)[0]) ==
                    #        str(bbmodel.predict(input_row)[0])):
                    #    print(explainer.predict(pert_row), bbmodel.predict(input_row))
                    #if use_slime:
                    #    assert False
                    temp_pert = None
                    temp_single = None
                    temp_double = None
                    fidelity_pert[i + j * local_test_end].append(temp_pert)
                    fidelity_single_blind[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind[i + j * local_test_end].append(temp_double)
                    fidelity_single_blind_e[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_e[i + j * local_test_end].append(temp_double)
                    fidelity_single_blind_n[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_n[i + j * local_test_end].append(temp_double)
                    # NEW
                    fidelity_single_diff_n[i + j * local_test_end].append(None)
                    fidelity_single_diff_e[i + j * local_test_end].append(None)
                    fidelity_double_diff_n[i + j * local_test_end].append(None)
                    fidelity_double_diff_e[i + j * local_test_end].append(None)
                    hit_rate[i + j * local_test_end].append(0)
                    avg_features[i + j * local_test_end].append(None)

            stability[i].append(calculate_stability(top_features))
            loose_stability[i].append(calculate_loose_stability(top_features, len(list(iris_training))))
            loose_stability5[i].append(calculate_loose_stability(top_features, 10))

    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate",
                       "Avg. Features",
                       "Stability 1", "Stab. 1 Var.", "Stability 2", "Stab. 2 Var.",
                       "Stability 3", "Stab. 3 Var.", "Stability 4", "Stab. 4 Var.",
                       "Stability 5", "Stab. 5 Var.",
                       "Loose Stab.", "Loose Stab. Var.",
                       "Loose Stab. 5", "Loose Stab. 5 Var."]]
    print(fidelity_pert)
    print(fidelity_single_blind)
    print(fidelity_double_blind)
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
                for k in range(10):
                    print("WHY ", fidelity_pert, i, k, j)
                    if fidelity_pert[i + k * local_test_end][j] is not None:
                        mean_count += 1
                        temp_pert_acc += fidelity_pert[i + k * local_test_end][j]
                        temp_single_acc += fidelity_single_blind[i + k * local_test_end][j]
                        temp_double_acc += fidelity_double_blind[i + k * local_test_end][j]
            temp_pert_std = 0
            temp_single_std = 0
            temp_double_std = 0
            for i in range(local_test_end):
                for k in range(10):
                    if fidelity_pert[i + k * local_test_end][j] is not None:
                        temp_pert_std += (fidelity_pert[i + k * local_test_end][j] - temp_pert_acc / mean_count) ** 2
                        temp_single_std += (fidelity_single_blind[i + k * local_test_end][
                                                j] - temp_single_acc / mean_count) ** 2
                        temp_double_std += (fidelity_double_blind[i + k * local_test_end][
                                                j] - temp_double_acc / mean_count) ** 2
            if mean_count == 1:
                temp_pert_std = 0
                temp_single_std = 0
                temp_double_std = 0
            else:
                temp_pert_std = (temp_pert_std / (mean_count - 1))
                temp_single_std = (temp_single_std / (mean_count - 1))
                temp_double_std = (temp_double_std / (mean_count - 1))
            # print(barbe_dist[j])
            # add standard deviations as info
            if (len(averages_print) - 1) / 3 <= j:
                averages_print.append([barbe_dist[j], "Perturbed"])
                averages_print.append([barbe_dist[j], "Single Blind"])
                averages_print.append([barbe_dist[j], "Double Blind"])
            if mean_count != 0:
                averages_print[(j * 3) + 1].append(temp_pert_acc / mean_count)
                averages_print[(j * 3) + 1].append(temp_pert_std)
                averages_print[(j * 3) + 2].append(temp_single_acc / mean_count)
                averages_print[(j * 3) + 2].append(temp_single_std)
                averages_print[(j * 3) + 3].append(temp_double_acc / mean_count)
                averages_print[(j * 3) + 3].append(temp_double_std)
            else:
                averages_print[(j * 3) + 1].append(0)
                averages_print[(j * 3) + 1].append(0)
                averages_print[(j * 3) + 2].append(0)
                averages_print[(j * 3) + 2].append(0)
                averages_print[(j * 3) + 3].append(0)
                averages_print[(j * 3) + 3].append(0)
    for j in range(len(barbe_dist)):
        average_hits = 0
        for i in range(local_test_end):
            for k in range(10):
                average_hits += hit_rate[i + k * local_test_end][j]
        average_hits /= local_test_end * 10
        averages_print[(j * 3) + 1].append(average_hits)
        averages_print[(j * 3) + 2].append(0)
        averages_print[(j * 3) + 3].append(0)
    for j in range(len(barbe_dist)):
        average_features = 0
        avg_count = 0
        for i in range(local_test_end):
            for k in range(10):
                if avg_features[i + k * local_test_end][j] is not None:
                    avg_count += 1
                    average_features += avg_features[i + k * local_test_end][j]
        if avg_count != 0:
            average_features /= avg_count
        averages_print[(j * 3) + 1].append(average_features)
        averages_print[(j * 3) + 2].append(-1)
        averages_print[(j * 3) + 3].append(-1)

    for k in range(5):
        for j in range(len(barbe_dist)):
            average_stab = 0
            stab_count = 0
            for i in range(local_test_end):
                if stability[i][j] is not None:
                    average_stab += stability[i][j][k]
                    stab_count += 1
            if stab_count != 0:
                average_stab /= stab_count
            averages_print[(j * 3) + 1].append(average_stab)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)
            stab_var = 0
            for i in range(local_test_end):
                if stability[i][j] is not None:
                    stab_var += (stability[i][j][k] - average_stab) ** 2
            if stab_count > 1:
                stab_var /= stab_count - 1
            averages_print[(j * 3) + 1].append(stab_var)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)

    for j in range(len(barbe_dist)):
        average_stab = 0
        stab_count = 0
        for i in range(local_test_end):
            if loose_stability[i][j] is not None:
                average_stab += loose_stability[i][j]
                stab_count += 1
        if stab_count != 0:
            average_stab /= stab_count
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            if loose_stability[i][j] is not None:
                stab_var += (loose_stability[i][j] - average_stab) ** 2
        if stab_count > 1:
            stab_var = (stab_var / (stab_count - 1))
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)

    for j in range(len(barbe_dist)):
        average_stab = 0
        stab_count = 0
        for i in range(local_test_end):
            if loose_stability5[i][j] is not None:
                average_stab += loose_stability5[i][j]
                stab_count += 1
        if stab_count != 0:
            average_stab /= stab_count
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            if loose_stability5[i][j] is not None:
                stab_var += (loose_stability5[i][j] - average_stab) ** 2
        if stab_count > 1:
            stab_var = (stab_var / (stab_count - 1))
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)


    pd.DataFrame(averages_print).to_csv("Results/lime_" + "_".join([data_name,
                                                               "nruns" + str(local_test_end),
                                                               "nperturb" + str(n_perturbations),
                                                               "barbe" + str(use_barbe_perturbations),
                                                               "devscalin" + str(dev_scaling)]) + "_results.csv")


def distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                            discrete_features=None,
                            local_test_end=20, data_name="data", n_perturbations=100, n_bins=5, dev_scaling=10, use_class_balance=False):
    # TODO: make this set up folds to run the experiments over and then be more generic
    random_seeds = [
                    117893, 7858801, 3256767, 98451, 787,
                    9631, 4482, 999999991, 30592, 123411,
                    117821213, 801, 3257, 51, 78,
                    31, 82, 99912, 3059, 411
                    ]
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

    importance_print = [["Metric", "Approach", "Distribution", "N Important", "Stability 5", "Stability N/2"]+
                        ["Feat 1 Stab", "Feat 2 Stab", "Feat 3 Stab", "Feat 4 Stab", "Feat 5 Stab"]+
                        [feat for feat in list(iris_wb_training)]]
    #barbe_dist = ['standard-normal', 'normal', 'uniform', 'cauchy', 't-distribution']
    if not use_class_balance:
        barbe_dist = ['standard-normal', 'normal']
    else:
        barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(local_test_end * 10)]
    fidelity_single_blind = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind = [[] for _ in range(local_test_end * 10)]
    # _e = euclidean, _n = nearest neighbors
    fidelity_single_blind_e = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind_e = [[] for _ in range(local_test_end * 10)]
    fidelity_single_blind_n = [[] for _ in range(local_test_end * 10)]
    fidelity_double_blind_n = [[] for _ in range(local_test_end * 10)]

    fidelity_single_diff_n = [[] for _ in range(local_test_end * 10)]
    fidelity_single_diff_e = [[] for _ in range(local_test_end * 10)]
    fidelity_double_diff_n = [[] for _ in range(local_test_end * 10)]
    fidelity_double_diff_e = [[] for _ in range(local_test_end * 10)]

    hit_rate = [[] for _ in range(local_test_end * 10)]
    avg_features = [[] for _ in range(local_test_end * 10)]
    stability = [[] for _ in range(local_test_end)]
    loose_stability = [[] for _ in range(local_test_end)]
    loose_stability5 = [[] for _ in range(local_test_end)]

    for i in range(local_test_end):  # use when removing LIME
        #i = 31  # loan 9 is female
        #i = 9
        pert_row = iris_perturb.iloc[i].copy()
        iris_wb_test = iris_test.drop(i, inplace=False, axis=0)
        for distribution in barbe_dist:
            top_features = [[] for _ in range(10)]
            for j in range(10):
                iris_wb_training = iris_training.sample(frac=1/3, random_state=random_seeds[j])
                try:
                    explainer = BARBE(training_data=iris_wb_training,
                                      input_bounds=None,#[(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                                      perturbation_type=distribution,
                                      n_perturbations=n_perturbations,
                                      dev_scaling_factor=dev_scaling,
                                      n_bins=n_bins,
                                      verbose=False,
                                      input_sets_class=False,
                                      balance_classes=use_class_balance)  # IAIN CHANGE BETWEEN FOR TESTS

                    splanation = explainer.explain(pert_row.copy(), bbmodel)
                    print("IAIN RULES: ", explainer.get_rules(applicable=pert_row.copy()))
                    print(pert_row)
                    print(splanation)
                    print(explainer._blackbox_classification['input'])
                    #if j > 8:
                    #    assert False
                    #print(explanation)
                    #print("IAIN DATA: ", explainer._perturbed_data)
                    #assert False
                    weight = None  # or nearest or euclidean
                    print("Explanation: ", j, " = ", splanation)
                    #import seaborn as sns
                    #import matplotlib.pyplot as plt

                    inflection_point = get_elbow(splanation)

                    #sns.scatterplot(x=[i for i in range(len(splanation))],
                    #                y=[j for _, j in splanation],
                    #                hue=['reg' if i != inflection_point else 'stop' for i in range(len(splanation))])
                    #plt.show()
                    #assert False

                    #splanation = [feature for feature, val in splanation]# if val > 0]
                    #splanation = splanation[0:inflection_point]

                    avg_features[i + j * local_test_end].append(inflection_point+1)

                    #print(splanation)
                    # print(len(splanation))
                    # assert False
                    # [["Metric", "Approach", "Distribution", "N Important", "Stability 5", "Stability N/2"]+
                    #                         ["Feat 1 Stab", "Feat 2 Stab", "Feat 3 Stab", "Feat 4 Stab", "Feat 5 Stab"]+
                    #                         [feat for feat in list(iris_wb_training)]]
                    feature_list = []
                    to_add_features = []
                    for feat_p in list(iris_wb_training):
                        found = False
                        for feat_i, imp in splanation:
                            #print(feat_i, feat_p)
                            if not found and feat_i == feat_p:
                                #print("Compared and added: ", imp)
                                feature_list.append(imp)
                                found = True
                        if not found:
                            feature_list.append(0)
                            to_add_features.append((feat_p, 0))
                    #assert False

                    #importance_print.append(["Old Approach", "BARBE", distribution, inflection_point+1, 0,
                    #                         0, 0, 0, 0, 0, 0] + feature_list)
                    random.shuffle(to_add_features)
                    top_features[j] = list(splanation.copy() + to_add_features)

                    temp_pert = explainer.get_surrogate_fidelity(comparison_model=bbmodel)
                    temp_single_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_training.drop('target', axis=1,
                                                                                                        errors='ignore'),
                                                                     weights=None,
                                                                     original_data=pert_row)
                    temp_double_f = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                     comparison_data=iris_wb_test.drop('target', axis=1,
                                                                                                       errors='ignore'),
                                                                     weights=None,
                                                                     original_data=pert_row)
                    fidelity_pert[i + j * local_test_end].append(temp_pert)
                    fidelity_single_blind[i + j * local_test_end].append(temp_single_f)
                    fidelity_double_blind[i + j * local_test_end].append(temp_double_f)
                    temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_training.drop('target', axis=1,
                                                                                                      errors='ignore'),
                                                                   weights='euclidean',
                                                                   original_data=pert_row)
                    temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_wb_test.drop('target', axis=1,
                                                                                                     errors='ignore'),
                                                                   weights='euclidean',
                                                                   original_data=pert_row)
                    fidelity_single_blind_e[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_e[i + j * local_test_end].append(temp_double)

                    fidelity_single_diff_e[i + j * local_test_end].append(temp_single - temp_single_f)
                    fidelity_double_diff_e[i + j * local_test_end].append(temp_double - temp_double_f)
                    temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_training,
                                                                   weights='nearest',
                                                                   original_data=pert_row)
                    temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel,
                                                                   comparison_data=iris_wb_test,
                                                                   weights='nearest',
                                                                   original_data=pert_row)
                    fidelity_single_blind_n[i + j * local_test_end].append(temp_single)
                    fidelity_double_blind_n[i + j * local_test_end].append(temp_double)

                    fidelity_single_diff_n[i + j * local_test_end].append(temp_single - temp_single_f)
                    fidelity_double_diff_n[i + j * local_test_end].append(temp_double - temp_double_f)
                    hit_rate[i + j * local_test_end].append(1)
                    #print(splanation)

                    del explainer
                    del splanation
                    #if len(top_features[j]) == 0:
                    #    print(top_features)
                    #    assert False

                except:
                    print(traceback.format_exc())
                    #assert False
                    print("Failed")
                    temp_pert = None
                    temp_single = None
                    temp_double = None
                    fidelity_pert[i+ j * local_test_end].append(temp_pert)
                    fidelity_single_blind[i+ j * local_test_end].append(temp_single)
                    fidelity_double_blind[i+ j * local_test_end].append(temp_double)
                    fidelity_single_blind_e[i+ j * local_test_end].append(temp_single)
                    fidelity_double_blind_e[i+ j * local_test_end].append(temp_double)
                    fidelity_single_blind_n[i+ j * local_test_end].append(temp_single)
                    fidelity_double_blind_n[i+ j * local_test_end].append(temp_double)
                    # NEW
                    fidelity_single_diff_n[i+ j * local_test_end].append(None)
                    fidelity_single_diff_e[i+ j * local_test_end].append(None)
                    fidelity_double_diff_n[i+ j * local_test_end].append(None)
                    fidelity_double_diff_e[i+ j * local_test_end].append(None)
                    hit_rate[i+ j * local_test_end].append(0)
                    avg_features[i + j * local_test_end].append(None)

                #print(i, " - ", distribution)
                #print(temp_pert)
                #print(temp_single)
                #print(temp_double)
            #[["Metric", "Approach", "Distribution", "N Important", "Stability 5", "Stability N/2"]+
            #                         ["Feat 1 Stab", "Feat 2 Stab", "Feat 3 Stab", "Feat 4 Stab", "Feat 5 Stab"]+
            #                         [feat for feat in list(iris_wb_training)]]

            stability[i].append(calculate_stability(top_features))
            loose_stability[i].append(calculate_loose_stability(top_features, len(list(iris_training))))
            loose_stability5[i].append(calculate_loose_stability(top_features, 10))
            #importance_print[1][4] = loose_stability5[i][-1]
            #importance_print[1][5] = loose_stability[i][-1]
            #if stability[i][-1] is not None:
            #    for d in range(0, 5):
            #        importance_print[1][d+6] = stability[i][-1][d]

            #pd.DataFrame(importance_print).to_csv("Results/old_sorting_all_rules_mult_normal_" + "_".join(["individual" + str(i),
            #                                                                         data_name,
            #                                                           "nruns" + str(local_test_end),
            #                                                           "nbins" + str(n_bins),
            #                                                           "nperturb" + str(n_perturbations),
            #                                                           "devscalin" + str(
            #                                                               dev_scaling)]) + "_results.csv")
            #assert False
            #print(calculate_loose_stability(top_features))
            #assert False
            #for feats in top_features:
            #    print(feats)
            #print(calculate_stability(top_features))
            #assert False
            #if calculate_stability(top_features) == 0:
            #    print(top_features)
            #    assert False

    # NEW
    averages_print = [["Method", "Evaluation",
                       "Fidelity (Original)", "Fid. Var.",
                       "Euclidean Fidelity", "Euc. Var.",
                       "Nearest Neighbor Fidelity", "NN. Var.",
                       "Euc. - Fidelity", "Euc. Diff. Var.",
                       "NN. - Fidelity", "NN. Diff. Var.", "Hit Rate",
                       "Avg. Features",
                       "Stability 1", "Stab. 1 Var.", "Stability 2", "Stab. 2 Var.",
                       "Stability 3", "Stab. 3 Var.", "Stability 4", "Stab. 4 Var.",
                       "Stability 5", "Stab. 5 Var.",
                       "Loose Stab.", "Loose Stab. Var.",
                       "Loose Stab. 5", "Loose Stab. 5 Var."]]
    print(fidelity_pert)
    print(fidelity_single_blind)
    print(fidelity_double_blind)
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
                for k in range(10):
                    print("WHY ", fidelity_pert, i, k, j)
                    if fidelity_pert[i + k * local_test_end][j] is not None:
                        mean_count += 1
                        temp_pert_acc += fidelity_pert[i + k * local_test_end][j]
                        temp_single_acc += fidelity_single_blind[i + k * local_test_end][j]
                        temp_double_acc += fidelity_double_blind[i + k * local_test_end][j]
            temp_pert_std = 0
            temp_single_std = 0
            temp_double_std = 0
            for i in range(local_test_end):
                for k in range(10):
                    if fidelity_pert[i + k * local_test_end][j] is not None:
                        temp_pert_std += (fidelity_pert[i + k * local_test_end][j] - temp_pert_acc / mean_count) ** 2
                        temp_single_std += (fidelity_single_blind[i + k * local_test_end][
                                                j] - temp_single_acc / mean_count) ** 2
                        temp_double_std += (fidelity_double_blind[i + k * local_test_end][
                                                j] - temp_double_acc / mean_count) ** 2
            if mean_count == 1:
                temp_pert_std = 0
                temp_single_std = 0
                temp_double_std = 0
            else:
                temp_pert_std = (temp_pert_std / (mean_count - 1))
                temp_single_std = (temp_single_std / (mean_count - 1))
                temp_double_std = (temp_double_std / (mean_count - 1))
            # print(barbe_dist[j])
            # add standard deviations as info
            if (len(averages_print) - 1) / 3 <= j:
                averages_print.append([barbe_dist[j], "Perturbed"])
                averages_print.append([barbe_dist[j], "Single Blind"])
                averages_print.append([barbe_dist[j], "Double Blind"])
            if mean_count != 0:
                averages_print[(j * 3) + 1].append(temp_pert_acc / mean_count)
                averages_print[(j * 3) + 1].append(temp_pert_std)
                averages_print[(j * 3) + 2].append(temp_single_acc / mean_count)
                averages_print[(j * 3) + 2].append(temp_single_std)
                averages_print[(j * 3) + 3].append(temp_double_acc / mean_count)
                averages_print[(j * 3) + 3].append(temp_double_std)
            else:
                averages_print[(j * 3) + 1].append(0)
                averages_print[(j * 3) + 1].append(0)
                averages_print[(j * 3) + 2].append(0)
                averages_print[(j * 3) + 2].append(0)
                averages_print[(j * 3) + 3].append(0)
                averages_print[(j * 3) + 3].append(0)
    for j in range(len(barbe_dist)):
        average_hits = 0
        for i in range(local_test_end):
            for k in range(10):
                average_hits += hit_rate[i + k * local_test_end][j]
        average_hits /= local_test_end * 10
        averages_print[(j * 3) + 1].append(average_hits)
        averages_print[(j * 3) + 2].append(0)
        averages_print[(j * 3) + 3].append(0)
    for j in range(len(barbe_dist)):
        average_features = 0
        avg_count = 0
        for i in range(local_test_end):
            for k in range(10):
                if avg_features[i + k * local_test_end][j] is not None:
                    avg_count += 1
                    average_features += avg_features[i + k * local_test_end][j]
        if avg_count != 0:
            average_features /= avg_count
        averages_print[(j * 3) + 1].append(average_features)
        averages_print[(j * 3) + 2].append(-1)
        averages_print[(j * 3) + 3].append(-1)

    for k in range(5):
        for j in range(len(barbe_dist)):
            average_stab = 0
            stab_count = 0
            for i in range(local_test_end):
                if stability[i][j] is not None:
                    average_stab += stability[i][j][k]
                    stab_count += 1
            if stab_count != 0:
                average_stab /= stab_count
            averages_print[(j * 3) + 1].append(average_stab)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)
            stab_var = 0
            for i in range(local_test_end):
                if stability[i][j] is not None:
                    stab_var += (stability[i][j][k] - average_stab) ** 2
            if stab_count > 1:
                stab_var /= stab_count - 1
            averages_print[(j * 3) + 1].append(stab_var)
            averages_print[(j * 3) + 2].append(0)
            averages_print[(j * 3) + 3].append(0)

    for j in range(len(barbe_dist)):
        average_stab = 0
        stab_count = 0
        for i in range(local_test_end):
            if loose_stability[i][j] is not None:
                average_stab += loose_stability[i][j]
                stab_count += 1
        if stab_count != 0:
            average_stab /= stab_count
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            if loose_stability[i][j] is not None:
                stab_var += (loose_stability[i][j] - average_stab) ** 2
        if stab_count > 1:
            stab_var = (stab_var / (stab_count - 1))
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)

    for j in range(len(barbe_dist)):
        average_stab = 0
        stab_count = 0
        for i in range(local_test_end):
            if loose_stability5[i][j] is not None:
                average_stab += loose_stability5[i][j]
                stab_count += 1
        if stab_count != 0:
            average_stab /= stab_count
        averages_print[(j * 3) + 1].append(average_stab)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
        stab_var = 0
        for i in range(local_test_end):
            if loose_stability5[i][j] is not None:
                stab_var += (loose_stability5[i][j] - average_stab) ** 2
        if stab_count > 1:
            stab_var = (stab_var / (stab_count - 1))
        averages_print[(j * 3) + 1].append(stab_var)
        averages_print[(j * 3) + 2].append(2)
        averages_print[(j * 3) + 3].append(2)
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
        if False:
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

    with open("./breast_cancer_neural_network.pickle", 'rb') as f:
        ptmodel = dill.load(f)

    for pert_c in [1000]:
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
        if True:
            lime_distribution_experiment(breast_cancer_training,
                                         breast_cancer_training_label,
                                         breast_cancer_perturb,
                                         breast_cancer_test,
                                         pre_trained_model=ptmodel,
                                         lime_version=LimeNewPert,
                                         local_test_end=50,
                                         data_name="SLIMETHISONE_RDONE_FINAL_fixed_missing_breast_cancer",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=False,
                                         use_slime=True,
                                         dev_scaling=1)
            #lime_distribution_experiment(breast_cancer_training,
            #                             breast_cancer_training_label,
            #                             breast_cancer_perturb,
            #                             breast_cancer_test,
            #                             pre_trained_model=ptmodel,
            #                             lime_version=LimeNewPert,
            #                             local_test_end=50,
            #                             data_name="slime_neural_stability_FINAL_fixed_missing_breast_cancer",
            #                             n_perturbations=1000,
            #                             use_barbe_perturbations=False,
            #                             use_slime=True,
            #                             dev_scaling=1)
        assert False
        lime_distribution_experiment(breast_cancer_training,
                                     breast_cancer_training_label,
                                     breast_cancer_perturb,
                                     breast_cancer_test,
                                     pre_trained_model=ptmodel,
                                     lime_version=LimeNewPert,
                                     local_test_end=50,
                                     data_name="BALLIN_DONE_fixed_missing_breast_cancer",
                                     n_perturbations=1000,
                                     use_barbe_perturbations=True,
                                     use_slime=False,
                                     dev_scaling=1)
            #assert False
        if False:
            baseline_distribution_experiment(breast_cancer_training,
                                    breast_cancer_training_label,
                                    breast_cancer_perturb,
                                    breast_cancer_test,
                                    # lime_version=LimeNewPert,
                                    pre_trained_model=ptmodel,
                                    local_test_end=50,
                                    data_name="stability_breast_cancer",
                                    n_perturbations=1000,
                                    # use_barbe_perturbations=False,
                                    use_class_balance=True,
                                    n_bins=10,
                                    dev_scaling=1)
        # IAIN reminder this needs to be rerun (there may be an error with dividing for lore)
        if False:
            lore_distribution_experiment(breast_cancer_training,
                                         breast_cancer_training_label,
                                         breast_cancer_perturb,
                                         breast_cancer_test,
                                         pre_trained_model=ptmodel,
                                         local_test_end=50,
                                         data_name="neural_stability_breast_cancer",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=False,
                                         dev_scaling=1)
        if True:
            for dev_n in [1]:#[1, 2, 3, 10, 50, 100]:
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
                #lime_distribution_experiment(breast_cancer_training,
                #                             breast_cancer_training_label,
                #                             breast_cancer_perturb,
                #                             breast_cancer_test,
                #                             lime_version=VAELimeNewPert,
                #                             local_test_end=50,
                #                             data_name="vaelime_neural_breast_cancer",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=False,
                #                             dev_scaling=dev_n)
                if True:
                    distribution_experiment(breast_cancer_training,
                                            breast_cancer_training_label,
                                            breast_cancer_perturb,
                                            breast_cancer_test,
                                            # lime_version=LimeNewPert,
                                            pre_trained_model=ptmodel,
                                            local_test_end=50,
                                            data_name="barbe_DONE_breast_cancer",
                                            n_perturbations=1000,
                                            # use_barbe_perturbations=False,
                                            use_class_balance=False,
                                            n_bins=10,
                                            dev_scaling=dev_n)
                    lore_distribution_experiment(breast_cancer_training,
                                                 breast_cancer_training_label,
                                                 breast_cancer_perturb,
                                                 breast_cancer_test,
                                                 pre_trained_model=ptmodel,
                                                 local_test_end=50,
                                                 data_name="neural_DONE_breast_cancer",
                                                 n_perturbations=1000,
                                                 use_barbe_perturbations=False,
                                                 dev_scaling=1)
                    if False:
                        distribution_experiment(breast_cancer_training,
                                                breast_cancer_training_label,
                                                breast_cancer_perturb,
                                                breast_cancer_test,
                                                # lime_version=LimeNewPert,
                                                pre_trained_model=ptmodel,
                                                local_test_end=50,
                                                data_name="barbe_DONE_stability_balanced_breast_cancer",
                                                n_perturbations=1000,
                                                # use_barbe_perturbations=False,
                                                use_class_balance=True,
                                                n_bins=10,
                                                dev_scaling=1)


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
    data = data.sample(frac=1, random_state=711)
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
    with open("./loan_neural_network.pickle", 'rb') as f:
        ptmodel = dill.load(f)

    for pert_c in [1000]:
        # lime_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=50,
        ##                             data_name="vaelime_breast_cancer",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if False:
            baseline_distribution_experiment(loan_training,
                                             loan_training_label,
                                             loan_perturb,
                                             loan_test,
                                             # lime_version=LimeNewPert,
                                             pre_trained_model=ptmodel,
                                             local_test_end=50,
                                             data_name="stability_loan",
                                             n_perturbations=1000,
                                             # use_barbe_perturbations=False,
                                             use_class_balance=True,
                                             n_bins=10,
                                             dev_scaling=1)
        if False:
            lime_distribution_experiment(loan_training,
                                         loan_training_label,
                                         loan_perturb,
                                         loan_test,
                                         pre_trained_model=ptmodel,
                                         lime_version=LimeNewPert,
                                         local_test_end=50,
                                         data_name="DONE_neuralstability_loan",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=False,
                                         dev_scaling=1)
        if False:
            lime_distribution_experiment(loan_training,
                                         loan_training_label,
                                         loan_perturb,
                                         loan_test,
                                         use_slime=False,
                                         pre_trained_model=ptmodel,
                                         lime_version=LimeNewPert,
                                         local_test_end=50,
                                         data_name="slime_DONE_neural_stability_loan",
                                         n_perturbations=2000,
                                         use_barbe_perturbations=False,
                                         dev_scaling=1)
        if False:
            lime_distribution_experiment(loan_training,
                                         loan_training_label,
                                         loan_perturb,
                                         loan_test,
                                         pre_trained_model=ptmodel,
                                         lime_version=LimeNewPert,
                                         local_test_end=50,
                                         data_name="DONE_neural_balance_stability_loan",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=True,
                                         dev_scaling=1)
        if True:
            for dev_n in [1]:  # [1, 2, 3, 10, 50, 100]:
                # lime_distribution_experiment(breast_cancer_training,
                #                             breast_cancer_training_label,
                #                             breast_cancer_perturb,
                #                             breast_cancer_test,
                #                             lime_version=LimeNewPert,
                #                             local_test_end=50,
                #                             data_name="bpert_neural_breast_cancer",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=True,
                #                             dev_scaling=dev_n)
                # lime_distribution_experiment(breast_cancer_training,
                #                             breast_cancer_training_label,
                #                             breast_cancer_perturb,
                #                             breast_cancer_test,
                #                             lime_version=VAELimeNewPert,
                #                             local_test_end=50,
                #                             data_name="vaelime_neural_breast_cancer",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=False,
                #                             dev_scaling=dev_n)
                if False:
                    distribution_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            # lime_version=LimeNewPert,
                                            pre_trained_model=ptmodel,
                                            local_test_end=50,
                                            data_name="barbe_DONE_neural_stability_loan",
                                            n_perturbations=1000,
                                            n_bins=10,
                                            use_class_balance=False,
                                            dev_scaling=dev_n)
                if True:
                    for i in range(0, 25, 5):
                        loan_perturb = data.iloc[i:i+5].reset_index()
                        loan_perturb = loan_perturb.drop(['index'], axis=1)
                        distribution_experiment(loan_training,
                                                loan_training_label,
                                                loan_perturb,
                                                loan_test,
                                                # lime_version=LimeNewPert,
                                                pre_trained_model=ptmodel,
                                                local_test_end=5,
                                                data_name="barbe_FDONE_items_"+str(i)+"_neural_stability_balance_loan",
                                                n_perturbations=1000,
                                                # use_barbe_perturbations=False,
                                                use_class_balance=True,
                                                n_bins=10,
                                                dev_scaling=1)
                if False:
                    lore_distribution_experiment(loan_training,
                                                 loan_training_label,
                                                 loan_perturb,
                                                 loan_test,
                                                 pre_trained_model=ptmodel,
                                                 local_test_end=50,
                                                 data_name="neural_stability_loan",
                                                 n_perturbations=1000,
                                                 # use_barbe_perturbations=False,
                                                 dev_scaling=1)
                #lime_distribution_experiment(loan_training,
                #                             loan_training_label,
                #                             loan_perturb,
                #                             loan_test,
                #                             pre_trained_model=ptmodel,
                #                             lime_version=LimeNewPert,
                #                             local_test_end=50,
                #                             data_name="neural_stability_elbow_balance_loan",
                ##                             n_perturbations=1000,
                #                             use_barbe_perturbations=True,
                #                             dev_scaling=1)


def test_distribution_experiment_aus_rain():
    data = pd.read_csv("../dataset/weatherAUS.csv")
    print(list(data))
    print(data.shape)
    # print(data.shape)
    data = data.drop(['Date', 'Location'], axis=1)
    data = data.dropna()
    print(data.shape)
    categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

    data = data.dropna()
    data = data.sample(frac=1, random_state=4670)
    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['RainTomorrow']:
            data[cat] = data[cat].astype(float)

    y = data['RainTomorrow']
    data = data.drop(['RainTomorrow'], axis=1)

    split_point_test = int((data.shape[0] * 0.4) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_training.drop('index', axis=1, inplace=True)

    loan_training_stop = int(loan_training.shape[0] // 3)
    for feature in list(loan_training):
        unique_values = np.unique(loan_training.iloc[0:loan_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
            loan_perturb[feature] = [(value if value in unique_values else "unknown") for value in
                                     loan_perturb[feature]]
            loan_test[feature] = [(value if value in unique_values else "unknown") for value in loan_test[feature]]
            loan_training[feature] = [(value if value in unique_values else "unknown") for value in
                                      loan_training[feature]]


    loan_perturb = loan_perturb.drop(['index'], axis=1)
    loan_test = loan_test.drop(['index'], axis=1)
    #loan_training = loan_training.drop(['index'], axis=1)
    #print(list(loan_test))
    #print(list(loan_perturb))
    #print(list(loan_training))
    #assert False
    with open("./aus_rain_neural_network.pickle", 'rb') as f:
        ptmodel = dill.load(f)

    for pert_c in [1000]:
        # lime_distribution_experiment(breast_cancer_training,
        #                             breast_cancer_training_label,
        #                             breast_cancer_perturb,
        #                             breast_cancer_test,
        #                             lime_version=VAELimeNewPert,
        #                             local_test_end=50,
        ##                             data_name="vaelime_breast_cancer",
        #                             n_perturbations=pert_c,
        #                             use_barbe_perturbations=False,
        #                             dev_scaling=1)
        if True:
            baseline_distribution_experiment(loan_training,
                                             loan_training_label,
                                             loan_perturb,
                                             loan_test,
                                             # lime_version=LimeNewPert,
                                             pre_trained_model=ptmodel,
                                             local_test_end=50,
                                             data_name="perturber_aus_rain",
                                             n_perturbations=1000,
                                             # use_barbe_perturbations=False,
                                             use_class_balance=True,
                                             n_bins=10,
                                             dev_scaling=1)
        if False:
            lime_distribution_experiment(loan_training,
                                         loan_training_label,
                                         loan_perturb,
                                         loan_test,
                                         pre_trained_model=ptmodel,
                                         lime_version=LimeNewPert,
                                         local_test_end=50,
                                         data_name="DONE_neural_stability_aus_rain",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=False,
                                         dev_scaling=1)
        if False:
            lime_distribution_experiment(loan_training,
                                         loan_training_label,
                                         loan_perturb,
                                         loan_test,
                                         pre_trained_model=ptmodel,
                                         lime_version=LimeNewPert,
                                         local_test_end=5,
                                         data_name="DONE_neural_balance_stability_aus_rain",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=False,
                                         dev_scaling=1)
        if True:
            for dev_n in [1]:  # [1, 2, 3, 10, 50, 100]:
                # lime_distribution_experiment(breast_cancer_training,
                #                             breast_cancer_training_label,
                #                             breast_cancer_perturb,
                #                             breast_cancer_test,
                #                             lime_version=LimeNewPert,
                #                             local_test_end=50,
                #                             data_name="bpert_neural_breast_cancer",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=True,
                #                             dev_scaling=dev_n)
                # lime_distribution_experiment(breast_cancer_training,
                #                             breast_cancer_training_label,
                #                             breast_cancer_perturb,
                #                             breast_cancer_test,
                #                             lime_version=VAELimeNewPert,
                #                             local_test_end=50,
                #                             data_name="vaelime_neural_breast_cancer",
                #                             n_perturbations=pert_c,
                #                             use_barbe_perturbations=False,
                #                             dev_scaling=dev_n)
                if False:
                    distribution_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            # lime_version=LimeNewPert,
                                            pre_trained_model=ptmodel,
                                            local_test_end=50,
                                            data_name="barbe_DONE_neural_stability_aus_rain",
                                            n_perturbations=1000,
                                            n_bins=10,
                                            use_class_balance=False,
                                            dev_scaling=dev_n)
                if False:
                    distribution_experiment(loan_training,
                                            loan_training_label,
                                            loan_perturb,
                                            loan_test,
                                            # lime_version=LimeNewPert,
                                            pre_trained_model=ptmodel,
                                            local_test_end=50,
                                            data_name="barbe_DONE_neural_stability_balance_aus_rain",
                                            n_perturbations=1000,
                                            # use_barbe_perturbations=False,
                                            use_class_balance=True,
                                            n_bins=10,
                                            dev_scaling=1)
                if False:
                    lore_distribution_experiment(loan_training,
                                                 loan_training_label,
                                                 loan_perturb,
                                                 loan_test,
                                                 pre_trained_model=ptmodel,
                                                 local_test_end=25,
                                                 data_name="neural_SECOND_HALF_DONE_aus_rain",
                                                 n_perturbations=1000,
                                                 # use_barbe_perturbations=False,
                                                 dev_scaling=1)
                if False:
                    lime_distribution_experiment(loan_training,
                                                 loan_training_label,
                                                 loan_perturb,
                                                 loan_test,
                                                 use_slime=False,
                                                 pre_trained_model=ptmodel,
                                                 lime_version=LimeNewPert,
                                                 local_test_end=25,
                                                 data_name="slime_DONE_neural_stability_aus_rain",
                                                 n_perturbations=2000,
                                                 use_barbe_perturbations=False,
                                                 dev_scaling=1)
                #lime_distribution_experiment(loan_training,
                #                             loan_training_label,
                #                             loan_perturb,
                #                             loan_test,
                #                             pre_trained_model=ptmodel,
                #                             lime_version=LimeNewPert,
                #                             local_test_end=50,
                #                             data_name="neural_stability_elbow_balance_loan",
                ##                             n_perturbations=1000,
                #                             use_barbe_perturbations=True,
                #                             dev_scaling=1)



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
    with open("./libras_neural_network.pickle", 'rb') as f:
        ptmodel = dill.load(f)

    for pert_c in [1000]: # [100, 500, 1000]:
        if False:
            baseline_distribution_experiment(libras_training,
                                    libras_training_label,
                                    libras_perturb,
                                    libras_test,
                                    pre_trained_model=ptmodel,
                                    #                        # lime_version=LimeNewPert,
                                    local_test_end=30,
                                    data_name="neural_libras",
                                    n_perturbations=1000,
                                    #                        # use_barbe_perturbations=False,
                                    n_bins=5,
                                    dev_scaling=1,
                                    use_class_balance=True)
        if False:
            lore_distribution_experiment(libras_training,
                                         libras_training_label,
                                         libras_perturb,
                                         libras_test,
                                         local_test_end=30,
                                         data_name="lore_neural_libras",
                                         n_perturbations=1000,
                                         use_barbe_perturbations=False,
                                         dev_scaling=1)
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
                if False:
                    lime_distribution_experiment(libras_training,
                                                 libras_training_label,
                                                 libras_perturb,
                                                 libras_test,
                                                 pre_trained_model=ptmodel,
                                                 lime_version=LimeNewPert,
                                                 local_test_end=30,
                                                 data_name="neural_libras",
                                                 n_perturbations=1000,
                                                 use_barbe_perturbations=False,
                                                 dev_scaling=dev_n)
                    lime_distribution_experiment(libras_training,
                                                 libras_training_label,
                                                 libras_perturb,
                                                 libras_test,
                                                 pre_trained_model=ptmodel,
                                                 lime_version=LimeNewPert,
                                                 local_test_end=30,
                                                 data_name="balanced_neural_libras",
                                                 n_perturbations=1000,
                                                 use_barbe_perturbations=True,
                                                 dev_scaling=dev_n)
                    lime_distribution_experiment(libras_training,
                                                 libras_training_label,
                                                 libras_perturb,
                                                 libras_test,
                                                 pre_trained_model=ptmodel,
                                                 lime_version=LimeNewPert,
                                                 local_test_end=30,
                                                 data_name="slime_neural_libras",
                                                 n_perturbations=1000,
                                                 use_barbe_perturbations=False,
                                                 use_slime=True,
                                                 dev_scaling=dev_n)
                ## IAIN running positive class only
                if True:
                    distribution_experiment(libras_training,
                                            libras_training_label,
                                            libras_perturb,
                                            libras_test,
                                            pre_trained_model=ptmodel,
                    #                        # lime_version=LimeNewPert,
                                            local_test_end=30,
                                            data_name="barbe_onlynextmost_neural_libras",
                                            n_perturbations=1000,
                    #                        # use_barbe_perturbations=False,
                                            n_bins=5,
                                           dev_scaling=dev_n)
                if True:
                    distribution_experiment(libras_training,
                                            libras_training_label,
                                            libras_perturb,
                                            libras_test,
                                            pre_trained_model=ptmodel,
                                            #                        # lime_version=LimeNewPert,
                                            local_test_end=30,
                                            data_name="barbe_onlynextmost_balanced_neural_libras",
                                            n_perturbations=1000,
                                            #                        # use_barbe_perturbations=False,
                                            n_bins=5,
                                            dev_scaling=dev_n,
                                            use_class_balance=True)
                if True:
                    lore_distribution_experiment(libras_training,
                                                 libras_training_label,
                                                 libras_perturb,
                                                 libras_test,
                                                 pre_trained_model=ptmodel,
                                                 local_test_end=30,
                                                 data_name="lore_neural_libras",
                                                 n_perturbations=1000,
                                                 use_barbe_perturbations=False,
                                                 dev_scaling=1)


from barbe.experiments.cross_validate import *


def test_breast_cancer_cv_model():
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

    cross_validate_nn(breast_cancer_training, breast_cancer_training_label, model_name="breast_cancer_neural_network")


def test_loan_cv_model():
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/train_loan_raw.csv")
    data = data.drop('Loan_ID', axis=1)
    data = data.dropna()
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

    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_training.drop('index', axis=1, inplace=True)

    loan_training_stop = int(loan_training.shape[0] // 3)
    for feature in list(loan_training):
        unique_values = np.unique(loan_training.iloc[0:loan_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
            loan_perturb[feature] = [(value if value in unique_values else "unknown") for value in loan_perturb[feature]]
            loan_test[feature] = [(value if value in unique_values else "unknown") for value in loan_test[feature]]
            loan_training[feature] = [(value if value in unique_values else "unknown") for value in loan_training[feature]]

    cross_validate_nn(loan_training, loan_training_label, model_name="loan_neural_network", use_categorical=categorical_features)


def test_libras_cv_model():
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/movement_libras.data",
                       names=([str(i) for i in range(1, 91)] + ['target']),
                       index_col=False)
    #print(data)
    #print(data.shape)
    #print(list(data))
    data = data[[str(i) for i in range(1, 91, 2)] + ['target']]
    data = data.sample(frac=1, random_state=117283)
    data.reset_index(drop=True, inplace=True)
    libtarget = [chr(y + 96) for y in data['target']]
    y = libtarget  # data['target'].astype(str)
    data = data.drop(['target'], axis=1)

    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    libras_perturb = data.iloc[0:30]
    libras_test = data.iloc[0:split_point_test]
    libras_training = data.iloc[split_point_test:]
    libras_training_label = y[split_point_test:]  # black box training labels

    cross_validate_nn(libras_training, libras_training_label, model_name="libras_neural_network")


def test_aus_rain_cv_model():
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/weatherAUS.csv")
    print(list(data))
    print(data.shape)
    #print(data.shape)
    data = data.drop(['Date', 'Location'], axis=1)
    data = data.dropna()
    print(data.shape)
    categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

    data = data.dropna()
    data = data.sample(frac=1, random_state=4670)
    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['RainTomorrow']:
            data[cat] = data[cat].astype(float)

    y = data['RainTomorrow']
    data = data.drop(['RainTomorrow'], axis=1)

    split_point_test = int((data.shape[0] * 0.4) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_training.drop('index', axis=1, inplace=True)

    loan_training_stop = int(loan_training.shape[0] // 3)
    for feature in list(loan_training):
        unique_values = np.unique(loan_training.iloc[0:loan_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
            loan_perturb[feature] = [(value if value in unique_values else "unknown") for value in loan_perturb[feature]]
            loan_test[feature] = [(value if value in unique_values else "unknown") for value in loan_test[feature]]
            loan_training[feature] = [(value if value in unique_values else "unknown") for value in loan_training[feature]]

    cross_validate_nn(loan_training, loan_training_label, model_name="aus_rain_neural_network", use_categorical=categorical_features)



#test_distribution_experiment_iris()
if __name__ == '__main__':
    #test_distribution_experiment_aus_rain()
    test_distribution_experiment_breast_cancer()
    #test_distribution_experiment_loan()


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
