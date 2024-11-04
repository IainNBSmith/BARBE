# IAIN make this file get and store results for each of the fidelity values we will check
#  against each other for each distribution type + lime's distribution

# experiments to see which of Lime, Uniform, Standard-Normal, Multi-Normal, Clustered-Normal, t-Distribution, Chauchy
#  perform the best in terms of fidelity values

# pert_fidelity = barbe[i].fidelity() -> training accuracy
# train_fidelity = barbe[i].fidelity(training_data, bbmodel) -> single blind / validation accuracy
# test_fidelity = barbe[i].fidelity(bbmodel_data, bbmodel) -> double blind / testing accuracy

from barbe.utils.lime_interface import LimeWrapper
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import barbe.tests.tests_config as tests_config
import random
from numpy.random import RandomState
from barbe.explainer import BARBE
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import pickle
'''
Remember the selling points of BARBE:
- More customizable than LIME
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
            # IAIN lime tends to produce data all with the same label
            temp_lime_pert_predictions[lime_pert_replace] = label
            temp_lime_single_predictions[lime_single_replace] = label
            temp_lime_double_predictions[lime_double_replace] = label
    return temp_lime_pert_predictions, temp_lime_single_predictions, temp_lime_double_predictions


def distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test, pre_trained_model=None,
                            discrete_features=None):
    # TODO: make this set up folds to run the experiments over and then be more generic
    random.seed(30)

    if pre_trained_model is None:
        bbmodel = RandomForestClassifier()
        bbmodel.fit(iris_training, iris_training_label)
    else:
        bbmodel = pre_trained_model

    #barbe_dist = ['normal', 'uniform', 'cauchy', 't-distribution']
    barbe_dist = ['normal']
    fidelity_pert = [[] for _ in range(20)]
    fidelity_single_blind = [[] for _ in range(20)]
    fidelity_double_blind = [[] for _ in range(20)]

    lime_fidelity_pert = [0 for _ in range(20)]
    lime_fidelity_single_blind = [0 for _ in range(20)]
    lime_fidelity_double_blind = [0 for _ in range(20)]

    lime_label_variety = [None for _ in range(20)]

    for i in range(20):
        pert_row = iris_perturb.iloc[i]
        #try:
        temp_lime_explainer = LimeTabularExplainer(training_data=iris_training.to_numpy(),
                                                   feature_names=list(iris_training),
                                                   discretizer='decile',
                                                   #training_labels=bbmodel.predict(iris_training),
                                                   discretize_continuous=False,
                                                   categorical_features=discrete_features,
                                                   categorical_names=discrete_features)
        _, lime_pert_data, lime_models = temp_lime_explainer.explain_instance(data_row=pert_row,
                                                                              predict_fn=bbmodel.predict_proba,
                                                                              labels=(0,1,),
                                                                              num_features=iris_training.shape[1],
                                                                              num_samples=5000,
                                                                              barbe_mode=False) # IAIN see if this works
        lime_pert_pred, lime_training_pred, lime_test_pred = (
            _lime_assess_predictions(lime_models, lime_pert_data, iris_training, iris_test))
        lime_pert_pred = np.array(lime_pert_pred)
        # IAIN leave as is until lime is fixed
        # TODO: tried lots of things but LIME just does not seem to perform well
        #lime_pert_pred[lime_pert_pred == 1] = 2
        #lime_pert_pred[lime_pert_pred == 0] = 1
        #print(lime_pert_pred.shape)
        #print(np.mean(lime_pert_pred))
        #assert False
        #print(len(bbmodel.predict(lime_pert_data)))
        lime_fidelity_pert[i] = accuracy_score(lime_pert_pred, bbmodel.predict(lime_pert_data))
        lime_fidelity_single_blind[i] = accuracy_score(lime_training_pred, bbmodel.predict(iris_training))
        lime_fidelity_double_blind[i] = accuracy_score(lime_test_pred, bbmodel.predict(iris_test))

        lime_label_variety[i] = lime_pert_pred
        #except:
        #    lime_fidelity_pert[i] = 0
        #    lime_fidelity_single_blind[i] = 0
        #    lime_fidelity_double_blind[i] = 0
        #    lime_label_variety[i] = 0
        for distribution in barbe_dist:

            try:
                iris_numpy = iris_training.to_numpy()
                explainer = BARBE(training_data=iris_training,
                                  input_bounds=None,#[(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                                  perturbation_type=distribution,
                                  n_perturbations=5000,
                                  dev_scaling_factor=5,
                                  n_bins=10,
                                  verbose=False,
                                  input_sets_class=False)

                explanation = explainer.explain(pert_row, bbmodel)

                temp_pert = explainer.get_surrogate_fidelity()
                temp_single = explainer.get_surrogate_fidelity(comparison_model=bbmodel, comparison_data=iris_training)
                temp_double = explainer.get_surrogate_fidelity(comparison_model=bbmodel, comparison_data=iris_test)
            except:
                temp_pert = None
                temp_single = None
                temp_double = None

            print(i, " - ", distribution)
            print(temp_pert)
            print(temp_single)
            print(temp_double)

            fidelity_pert[i].append(temp_pert)
            fidelity_single_blind[i].append(temp_single)
            fidelity_double_blind[i].append(temp_double)

    print(fidelity_pert)
    print(fidelity_single_blind)
    print(fidelity_double_blind)

    for j in range(len(barbe_dist)):
        temp_pert_acc = 0
        temp_single_acc = 0
        temp_double_acc = 0
        mean_count = 0
        for i in range(20):
            if fidelity_pert[i][j] is not None:
                mean_count += 1
                temp_pert_acc += fidelity_pert[i][j]
                temp_single_acc += fidelity_single_blind[i][j]
                temp_double_acc += fidelity_double_blind[i][j]
                #print("LIME Label Variety")
                #print(np.mean(lime_label_variety[i]))

        print(barbe_dist[j])
        if mean_count != 0:
            print("Fidelity: ", temp_pert_acc/mean_count)
            print("LIME: ", np.mean(lime_fidelity_pert))
            print("Single Blind: ", temp_single_acc/mean_count)
            print("LIME: ", np.mean(lime_fidelity_single_blind))
            print("Double Blind: ", temp_double_acc/mean_count)
            print("LIME: ", np.mean(lime_fidelity_double_blind))
    print([(np.nanmin(iris_numpy[:,0]), np.nanmax(iris_numpy[:,0])),
                                                (np.nanmin(iris_numpy[:,1]), np.nanmax(iris_numpy[:,1])),
                                                (np.nanmin(iris_numpy[:,2]), np.nanmax(iris_numpy[:,2])),
                                                (np.nanmin(iris_numpy[:,3]), np.nanmax(iris_numpy[:,3])),])


def distribution_experiment_iris():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df = iris_df.sample(frac=1).reset_index(drop=True)

    iris_perturb = iris_df.iloc[0:20]
    iris_test = iris_df.iloc[20:60]
    iris_training = iris_df.iloc[60:]
    iris_training_label = iris.target[60:]

    distribution_experiment(iris_training, iris_training_label, iris_perturb, iris_test)


def distribution_experiment_loan():
    # example of where lime fails
    # lime can only explain pre-processed data (pipeline must be separate and interpretable from model)
    data = pd.read_csv("../dataset/train_loan_raw.csv", index_col=0)
    print(list(data))
    data = data.dropna()
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    data = data.dropna()
    data = data.sample(frac=1)
    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Loan_Status']:
            data[cat] = data[cat].astype(float)

    y = data['Loan_Status']
    data = data.drop(['Loan_Status'], axis=1)

    preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    model = Pipeline([('pre', preprocess),
                      ('clf', RandomForestClassifier())])

    loan_perturb = data.iloc[0:20]
    loan_test = data.iloc[20:60]
    loan_training = data.iloc[60:]
    loan_training_label = y[60:]

    model.fit(loan_training, loan_training_label)

    distribution_experiment(loan_training, loan_training_label, loan_perturb, loan_test, pre_trained_model=model,
                            discrete_features=categorical_features)


distribution_experiment_loan()
