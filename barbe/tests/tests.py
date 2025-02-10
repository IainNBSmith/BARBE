"""
This code contains tests that ensure the BARBE package is working correctly.
"""
import lime1.lime_tabular
from VAELIME.lime2.lime_tabular import *
from barbe.utils.lime_interface import LimeNewPert, VAELimeNewPert

'''
    (COMPLETE/July 6th) TODO: Get SigDirect to be able to compile
    (COMPLETE/July 6th) TODO: Incorporate SigDirect into the explain function (pass the data)
    (COMPLETE/July 7th) TODO: Check explanation coming out of SigDirect method for correctness
        - Problem is explanations do not yield correct predictions atm (check if input data matches)
    (COMPLETE/July 7th -> July 8th) TODO: Compare results to original writing of BARBE
        - Explanation is the same as the original barbe function I ran, so this seems complete
    (COMPLETE/July 9th) TODO: Get BARBE to run only with the lime_tabular inverse function
    (COMPLETE/July 9th) TODO: Check rules from the SigDirect or BARBE paper (whatever used GLASS)
    (COMPLETE/July 9th) TODO: Get SigDirect working properly
        - Changed settings in SigDirect and it started to output correct classifications
        - Now we offer the setting of using all classes or just a binary classification for the given row's class
    (July 9th -> July 10th) TODO: Try different input data with named columns maybe
    (WORKED ON/July 10th -> July 11th) TODO: Clean up the written program
    (NOTE/July 13th) - Encoder is modified, but support ends up inconsistent so we need a fix
    (NOTE/July 13th) - Check why fidelity can be one yet it does not provide any rules
    (COMPLETE/July 16th) TODO: Make your own perturber
        - Needed to scale back the standard deviation to improve the results
    (July 11th -> July 17th) TODO: Start work on paper review (look into paper share by Osmar)
    (July 11th -> July 17th) TODO: Ensure that data format of the input row is correct for both types of data
        - Include in tests.py (a test where we check this works)
    (COMPLETE/July 11th -> July 17th) TODO: Setup data to rerun a few times until the classification is correct
        - Easy to include a check in the called function
    (COMPLETE/July 15th -> July 17th) TODO: Return the exact rules that were used on the case (e.g. 2<'sepal width (cm)'<5)
    (COMPLETE/July 19th) TODO: Add alternative to training data for perturbing data e.g. categorical_info, feature_names, scales or bounds (switches perturbation_mode)
    (Aug 21st) TODO: Add loans data set test and saves
    (Aug 21st) TODO: Add names to glass dataset
    (Aug 22nd) TODO: Perturb categorical as one hot values or find a different method
    '''
from barbe.utils.lime_interface import LimeNewPert
from datetime import datetime
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import barbe.tests.tests_config as tests_config
import random
from numpy.random import RandomState
from barbe.explainer import BARBE
import numpy as np

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

from barbe.perturber import BarbePerturber

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
# np.random.seed(seed=RANDOM_SEED)
const_random_state = RandomState(RANDOM_SEED)


# global variables
shrink_train_size = True
desired_train_size = 100

shrink_test_size = True
desired_test_size = 100

# used if data is not already split
TRAIN_RATIO = 0.75

PROCESS_COUNT = 50

# compute averages over all instances, or over the ones that the labels agree
fidelity_division = True

# how many runs to make sure fidelity is respected in BARBE/XLIME
REPEAT_COUNT = 1

#DATA_ROOT_DIR = experiments_config.DATA_ROOT_DIR
DATA_ROOT_DIR = "../dataset"

datasets_info_dict = tests_config.all_datasets_info_dict.copy()

datasets_info_dict = {'glass': tests_config.all_datasets_info_dict['glass']}


info = {'max_explanation_size':5}

remove_datasets = [
    'online_shoppers_intention', # slow with 1k!
    'breast-cancer', # very slow even in 200!, but good results!
    'car', # has a small tree! f1=55, fidel:82
    'nursery', # precision=1, recall=.5
    'adult', # slow, 1k won't finish in 2h
]
# remove_datasets = ['online_shoppers_intention']


def _get_train_df(filename, has_index, header_index=None):
    print("Entered Train DF: ", filename, os.getcwd())
    df = pd.read_csv(filename, sep=',', header=header_index, na_values='?')
    print("Loaded DF")
    if has_index:
        df = df.drop(df.columns[0], axis=1)
    return df


def _preprocess_data(train_df, class_index, dataset_name):
    # removing class label, so we can call get_dummies on the rest
    train_df.rename(columns={list(train_df.columns)[class_index]: 'class'}, inplace=True)
    print(train_df)
    print({list(train_df.columns)[class_index]: 'class'})
    train_class = train_df['class']
    print(train_class)
    train_df.drop(columns=['class'], inplace=True)
    print(train_df)

    # test_df.rename(columns={list(test_df.columns)[class_index]: 'class'}, inplace=True)
    # test_class = test_df['class']
    # test_df.drop(columns=['class'], inplace=True)

    # process categorical data
    infer_types = []
    #
    # df = pd.concat([train_df, test_df])  # Now they no class column
    df = train_df

    for column in df.columns:
        if df[column].dtype == 'object':
            infer_types.append("{}_CAT".format(column))
        else:
            infer_types.append("{}_NUM".format(column))
    datasets_info_dict[dataset_name]['_NUM'] = sum(['_NUM' in x for x in infer_types])
    datasets_info_dict[dataset_name]['_CAT'] = sum(['_CAT' in x for x in infer_types])
    #     print('INFER_TYPES:', infer_types)

    df = pd.get_dummies(df)
    train_df = df[:train_df.shape[0]]
    test_df = df[train_df.shape[0]:]

    print(train_df)
    print(test_df)
    assert set(train_df.columns) == set(test_df.columns)

    # process numerical data (standardization is independent[?] for train/test splits)
    continuous_column_names = [x for x in list(train_df.columns) if not '_' in str(x)]
    print(continuous_column_names)
    for column in continuous_column_names:
        # standardazing the column
        scaler = StandardScaler()
        train_df[column] = scaler.fit_transform(train_df[column].to_numpy().reshape((-1, 1)))
        # test_df[column] = scaler.transform(test_df[column].to_numpy().reshape((-1, 1)))

        # set NaN to 0
        train_df[column].fillna(0., inplace=True)
        # test_df[column].fillna(0., inplace=True)

    train_df['class'] = train_class
    # test_df['class'] = test_class

    print(train_df)
    print(test_df)
    # return train_df, test_df
    return train_df, None


def _get_data():
    for dataset, dataset_info in datasets_info_dict.items():
        print('---', dataset)
        print('---', dataset_info)
    dataset_name = dataset
    dataset_info = dataset_info
    class_index = dataset_info['CLASS_INDEX']
    has_index = dataset_info['HAS_INDEX']
    header_index = dataset_info['HEADER_ROW_NUMBER']
    # dataset_name, dataset_info
    filenames = ['../dataset/glass.data']  # IAIN

    # load from file
    df = _get_train_df(filenames[0], has_index, header_index)
    # shuffle
    df = df.sample(frac=1, random_state=const_random_state)

    train_size = int(df.shape[0] * TRAIN_RATIO)
    train_df = df.iloc[:train_size]

    # some of the preprocessing (one hot encoding + missing values + standardisation)
    dataset_info['initial_features'] = train_df.shape[1] - 1

    train_df, _ = _preprocess_data(train_df, class_index, dataset_name)

    dataset_info['features'] = train_df.shape[1] - 1
    dataset_info['initial_train_size'] = train_df.shape[0]
    # dataset_info['initial_test_size'] = test_df.shape[0]
    dataset_info['original_train_df'] = train_df.copy()
    # dataset_info['original_test_df'] = test_df.copy()

    if shrink_train_size and train_df.shape[0] > desired_train_size:
        train_df = train_df[:desired_train_size]
    # if shrink_test_size and test_df.shape[0] > desired_test_size:
    #     test_df = test_df[:desired_test_size]
    dataset_info['train_size'] = train_df.shape[0]
    # dataset_info['test_size'] = test_df.shape[0]

    # return train_df, test_df
    return train_df, None


def test_produce_lime_perturbations(n_perturbations=5000):
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed

    training_data, _ = _get_data()
    data_row = training_data.drop('class', axis=1).iloc[0]

    print("Running test: Lime Perturbation")
    start_time = datetime.now()
    lw = LimeWrapper(training_data.drop('class', axis=1), training_data['class'])
    perturbed_data = lw.produce_perturbation(data_row, n_perturbations)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(perturbed_data)


def test_documentation():
    help(BARBE)


def test_simple_numeric():
   pass


def test_simple_text():
    pass


def test_iris_dataset():
    # TODO: bring this test to Osmar
    iris = datasets.load_iris()
    training_data = iris.data

    temp_lime_explainer = (
        VAELimeNewPert(training_data=iris.data,
                                               feature_names=iris.feature_names,
                                               discretizer='quartile',
                                               # training_labels=bbmodel.predict(iris_training),
                                               discretize_continuous=False,
                                               categorical_features=[],
                                               categorical_names=[]))
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    data_row = iris_df.iloc[10]  # 50 got very wrong, 10 is good
    training_labels = iris.target
    training_data = iris_df
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)
    print('BBMODEL')
    print('Feature Importance: ', bbmodel.feature_importances_)
    print(bbmodel.score(training_data, training_labels))
    print(confusion_matrix(training_labels, bbmodel.predict(training_data)))
    lime_explanations = temp_lime_explainer.explain_instance(data_row=data_row,
                                                                          predict_fn=bbmodel.predict_proba,
                                                                          labels=(0, 1, 2),
                                                                          num_features=iris.data.shape[1],
                                                                          num_samples=5000)
    #lime_explanations.show_in_notebook(show_table=True, show_all=False)
    print('LIME')
    print(iris.feature_names)
    print('Feature Importance: ', lime_explanations.as_list())
    lime_pert_data = pd.DataFrame(temp_lime_explainer.perturbed_data, columns=[str(i) for i in range(4)])
    print(confusion_matrix(bbmodel.predict(temp_lime_explainer.perturbed_data),
                           temp_lime_explainer.predict(lime_pert_data)))
    print(confusion_matrix(bbmodel.predict(training_data), temp_lime_explainer.predict(training_data)))
    print(bbmodel.predict(data_row.to_numpy().reshape(1,-1)))
    print(bbmodel.predict_proba(data_row.to_numpy().reshape(1, -1)))
    new_data = temp_lime_explainer.scale_data(data_row.to_numpy().reshape(1, -1))[0]
    print(new_data)
    for label in temp_lime_explainer.base.local_model.keys():
        print(label, ": ", temp_lime_explainer.base.local_model[label].predict(new_data[temp_lime_explainer.base.used_features[label]].reshape(1, -1)))
    #assert False
    print("Running test: BARBE iris Run")
    start_time = datetime.now()
    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data,
                      verbose=False,
                      input_sets_class=False,
                      perturbation_type='normal',
                      dev_scaling_factor=2)
    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print('BARBE')
    #self._blackbox_classification = {'input': None, 'perturbed': None}
    #self._surrogate_classification = {'input': None, 'perturbed': None}
    print(confusion_matrix(explainer._blackbox_classification['perturbed'],
                           explainer._surrogate_classification['perturbed']))
    print(confusion_matrix(bbmodel.predict(training_data).astype(str),
                           explainer._surrogate_model.predict(training_data)))
    print(data_row)
    print(list(training_data))
    print('Feature Importance: ', explanation)
    print("ALL RULES:", explainer.get_rules())
    # example modification
    # IAIN TODO: check in valid range the full edges do not matter
    data_row['sepal length (cm)'] = -10
    print("DATA:", data_row)
    print("CONTRAST:", explainer.get_contrasting_rules(data_row))


def test_iris_give_scale_category():
    # IAIN add traffic dataset to this work with trained classification being
    #  the day of the week?
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # IAIN explain breaks for IRIS why??
    data_row = iris_df.iloc[50]
    training_labels = iris.target
    training_data = iris_df

    print("Running test: BARBE iris Run")
    start_time = datetime.now()
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)

    training_data.to_csv("../dataset/iris_test.csv")
    with open("../pretrained/iris_test_decision_tree.pickle", "wb") as f:
        pickle.dump(bbmodel, f)

    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=True, input_sets_class=True)
    print("INPUT PREMADE PERTURBATION INFO")
    print("SCALE FROM OTHER:", explainer.get_perturber(feature='scale'))
    print("SCALE USED IF DIFFERENT:", [0.2, 0.1, 0.5, 0.1])
    print("CATEGORIES:", explainer.get_perturber(feature='categories'))
    explainer = BARBE(input_scale=[0.2, 0.1, 1e-9, 0.1],
                      input_categories=explainer.get_perturber(feature='categories'),
                      feature_names=list(training_data),
                      verbose=True, input_sets_class=True)
    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(explanation)
    print(bbmodel.feature_importances_)
    print("ALL RULES:", explainer.get_rules())
    # example modification
    # IAIN TODO: check in valid range the full edges do not matter
    data_row['sepal length (cm)'] = -10
    print("DATA:", data_row)
    print("PERTURBED DATA: ", explainer.get_perturbed_data())
    print("CONTRAST:", explainer.get_contrasting_rules(data_row))


def test_iris_give_bounds():
    # IAIN add traffic dataset to this work with trained classification being
    #  the day of the week?
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # IAIN explain breaks for IRIS why??
    data_row = iris_df.iloc[50]
    training_labels = iris.target
    training_data = iris_df

    print("Running test: BARBE iris Run")
    start_time = datetime.now()
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)

    training_data.to_csv("../dataset/iris_test.csv")
    with open("../pretrained/iris_test_decision_tree.pickle", "wb") as f:
        pickle.dump(bbmodel, f)

    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=True, input_sets_class=True)
    print("INPUT PREMADE PERTURBATION INFO")
    print("SCALE FROM OTHER:", explainer.get_perturber(feature='scale'))
    print("SCALE USED IF DIFFERENT:", [0.2, 0.1, 0.5, 0.1])
    print("CATEGORIES:", explainer.get_perturber(feature='categories'))
    explainer = BARBE(input_scale=[0.2, 100, 100, 100],
                      input_categories=explainer.get_perturber(feature='categories'),
                      input_bounds=[None, (None, 0.1), (0, None), (0, 1)],
                      perturbation_type='t-distribution',
                      feature_names=list(training_data),
                      verbose=True, input_sets_class=True)
    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(explanation)
    print(bbmodel.feature_importances_)
    print("ALL RULES:", explainer.get_rules())
    # example modification
    # IAIN TODO: check in valid range the full edges do not matter
    data_row['sepal length (cm)'] = -10
    print("DATA:", data_row)
    print("PERTURBED DATA: ", explainer.get_perturbed_data())
    print("CONTRAST:", explainer.get_contrasting_rules(data_row))


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
def test_loan():
    data = pd.read_csv("../dataset/train_loan_raw.csv", index_col=0)
    print(list(data))
    data = data.dropna()
    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    data = data.dropna()
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

    model.fit(data, y)
    data.to_csv("../dataset/loan_test.csv")
    with open("../pretrained/loan_test_decision_tree.pickle", "wb") as f:
        pickle.dump(model, f)
    print(confusion_matrix(model.predict(data), y))
    print(model.predict(data))


def test_loan_open():
    # TODO: see why crashes happen in webpage but not here
    #  TODO: based on results here I think it is a formatting innaccury, should pass categorical info from the perturber to be certain
    data = pd.read_csv("../dataset/train_loan_raw.csv", index_col=0)
    print(data.dtypes)
    data = data.dropna()
    y = data['Loan_Status']
    data = data.drop(['Loan_Status'], axis=1)
    with open("../pretrained/loan_test_decision_tree.pickle", "rb") as f:
        model = pickle.load(f)

    explainer = BARBE(training_data=data, input_categories=None, verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')
    print("HOW FROM EXPLAINER: ", explainer.get_perturber('categories'))
    cov = explainer.get_perturber('covariance')
    np.fill_diagonal(cov, explainer.get_perturber('scale')**0.5)
    # Lots of tuning to get the right mix of scale value modifications and perturber
    #  should run tests with all data to see if there is a trend and we can avoid this step
    explainer = BARBE(training_data=None, input_categories=explainer.get_perturber('categories'),
                      input_scale=(explainer.get_perturber('scale')**2.5)/1,
                      input_means=explainer.get_perturber('means'),
                      feature_names=list(data),
                      input_covariance=cov/100,
                      verbose=True, input_sets_class=False,
                      higher_frequent_category_odds=True,
                      #dev_scaling_factor=0.1,
                      perturbation_type='normal',
                      n_perturbations=500)
    # TODO: consider only dividing the variation on categorical values
    # TODO: only normal distribution seems to work, see what the other ones do...
    # TODO: also try to standardize the variation and see if that works better in general
    # explainer.get_perturber('scale')**0.5 works for t-distribution (and works), so it is feature dependent
    # **0.1 worked for uniform
    explanation = explainer.explain(data.iloc[0], model)
    print("EVALUATION")
    print("CONFUSION: ", confusion_matrix(model.predict(data), y))
    print("PREDICTIONS: ", model.predict(data))
    a = explainer.get_surrogate_fidelity()
    b = explainer.get_surrogate_fidelity(comparison_data=data, original_data=data.iloc[0], comparison_model=model,
                                         weights='nearest-neighbors')
    b2 = explainer.get_surrogate_fidelity(comparison_data=data, original_data=data.iloc[0], comparison_model=model,
                                         weights='euclidean')
    b3 = explainer.get_surrogate_fidelity(comparison_data=data, original_data=data.iloc[0], comparison_model=model)
    c = confusion_matrix(y, explainer._surrogate_model.predict(data))
    print("CONFUSION DATA: \n", confusion_matrix(model.predict(data), y))
    print("CONFUSION SURROGATE DATA: \n", c)
    print("CONFUSION PERTURBED: \n", confusion_matrix(explainer._blackbox_classification['perturbed'],
                                            explainer._surrogate_classification['perturbed']))
    # pert, original, euclidean, nearest
    print("FIDELITY: ", a, b3, b2, b)
    # print(np.unique(explainer._perturbed_data[:, 0]))


def test_iris_counterfactual():
    # IAIN add traffic dataset to this work with trained classification being
    #  the day of the week?
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    data_row = iris_df.iloc[51]
    training_labels = iris.target
    training_data = iris_df

    print("Running test: BARBE iris Counterfactual")
    start_time = datetime.now()
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)
    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=True, input_sets_class=False,
                      dev_scaling_factor=3, perturbation_type='normal', n_bins=10)
    explanation = explainer.explain(data_row, bbmodel)
    counterfactual = explainer.get_counterfactual_explanation(data_row, wanted_class='0', n_counterfactuals=5)
    print("Test Time: ", datetime.now() - start_time)
    print(explainer.get_surrogate_fidelity())
    print("APPLIED RULES: ", explainer.get_rules(applicable=data_row))
    print(data_row)
    print(counterfactual[0])
    print(counterfactual[1])
    print("BARBE Expected Classes: ", counterfactual[2])
    print("Original Class: ", bbmodel.predict(data_row.to_numpy().reshape(1, -1)))
    print(bbmodel.predict(counterfactual[0][0]))
    print(bbmodel.predict(counterfactual[0][1]))
    print(bbmodel.predict(counterfactual[0][2]))

def test_glass_dataset():
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed

    training_data, _ = _get_data()
    training_data.columns = training_data.columns.astype(str)
    # IAIN something is going on I think with the input data (have to check tomorrow)
    #  something weird is going on with the rules being translated/used. It could have to
    #  do with something improper in the predictions (check that multiple lables are given)
    #
    # To try to fix this I have instead (and think the perturber needs) called the barbe
    #  version of perturbations which returns a OneHotEncoder (I think this can be made separately)
    #  this means in BARBE I have an element called self._why which holds the sd_data that seems to be
    #  used by SigDirect as the perturbed data. What I need to do tomorrow is check that the labels have some
    #  significance and then compare the data at each step to data in the current barbe.py file.
    # IAIN IMPORTANT: now as written it yields rules so this only has to be cleaned up before starting your own encoder
    #  you should also check with data that has named columns too (seems to always yield the same rule though)
    #  I assume this is because it is not actually using the one given row, it is using the zeroth row of the perturbed
    #  data. So I need to fix this to ensure rules are accurate, need to find out how though.
    #  (can the ohe encode new data?)
    # IAIN TODO: check the sets that are passed to ensure that the classes are correct as it seems to flip-flop classes
    #  for example 2's are 7's tho I wonder why the accuracy is still so high with this
    # data_row = training_data.drop('class', axis=1).loc[training_data['class'] == 7].iloc[10]  # IAIN most recent change yields more rules
    # print("Feature corruption")
    data_row = training_data.drop('class', axis=1).iloc[10]
    training_labels = training_data['class']
    training_data = training_data.drop('class', axis=1)
    # For overfit
    # data_row = training_data.iloc[5]
    # training_labels = training_data['class']

    print("Running test: BARBE Glass Run")
    start_time = datetime.now()
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)

    training_data.to_csv("../dataset/glass_test.csv")
    with open("../pretrained/glass_test_decision_tree.pickle", "wb") as f:
        pickle.dump(bbmodel, f)
    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=False, input_sets_class=True,
                      perturbation_type='uniform', dev_scaling_factor=5)
    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(explanation)
    print(explainer.get_categories())
    print(bbmodel.feature_importances_)
    all_rules = explainer.get_rules()
    print("ALL RULES:", all_rules)
    print(len(all_rules))
    contrast_rules = explainer.get_contrasting_rules(data_row)
    print("CONTRAST:", contrast_rules)
    print(len(contrast_rules))
    # explainer.get_rules()


def test_glass_counterfactual():
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed

    training_data, _ = _get_data()
    training_data.columns = training_data.columns.astype(str)
    data_row = training_data.drop('class', axis=1).iloc[0]
    training_labels = training_data['class']
    training_data = training_data.drop('class', axis=1)
    # For overfit
    # data_row = training_data.iloc[5]
    # training_labels = training_data['class']

    print("Running test: BARBE Glass Counterfactual")
    start_time = datetime.now()
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)

    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=False, input_sets_class=False,
                      perturbation_type='normal', dev_scaling_factor=5)
    explanation = explainer.explain(data_row, bbmodel)
    counterfactual = explainer.get_counterfactual_explanation(data_row, wanted_class='7')
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(counterfactual[0])
    print(counterfactual[2])
    print(bbmodel.predict(data_row.to_numpy().reshape(1,-1)))
    print(bbmodel.predict(counterfactual[0]))
    print(explainer.get_surrogate_fidelity())
    # explainer.get_rules()


def test_loan_counterfactual():
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed

    training_data = pd.read_csv("../dataset/loan_test.csv", index_col=0)
    data_row = training_data.iloc[0]
    # For overfit
    # data_row = training_data.iloc[5]
    # training_labels = training_data['class']

    print("Running test: BARBE Glass Counterfactual")
    start_time = datetime.now()
    with open('../pretrained/loan_test_decision_tree.pickle', 'rb') as file:
        bbmodel = pickle.load(file)

    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=False, input_sets_class=False,
                      perturbation_type='normal', dev_scaling_factor=1)
    input_categories = explainer.get_perturber(feature='categories')
    print(input_categories)
    print(data_row)
    input_categories[0] = {'Male': 1}
    input_categories[1] = {'Yes': 1}
    input_categories[2] = {'1': 1}
    input_categories[9] = {'1.0': 1}

    input_scale = explainer.get_perturber(feature='scale') / 2
    #input_bounds = explainer.get_
    # TODO: find out why there is not any agreement make sure that we trust it being wrong
    explainer = BARBE(input_scale=input_scale, input_bounds=[None, None, None, None, None, [4000, 10000], [0, 2000],
                                                             None, None, None, None],
                      input_categories=input_categories, feature_names=list(training_data),
                      verbose=False, input_sets_class=False,
                      perturbation_type='normal', n_bins=5, n_perturbations=5000)
    explanation = explainer.explain(data_row, bbmodel)
    counterfactual = explainer.get_counterfactual_explanation(data_row, wanted_class='Y', n_counterfactuals=3)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(counterfactual[0])
    print(counterfactual[1])
    print(counterfactual[2])
    pd_og = pd.DataFrame(columns=list(training_data))
    pd_og.loc[0] = data_row
    print(bbmodel.predict(pd_og))
    #print(counterfactual)
    pd_cf = pd.DataFrame(columns=list(training_data))
    pd_cf.loc[0] = counterfactual[0][0][0]
    print(bbmodel.predict(pd_cf))
    pd_cf.loc[0] = counterfactual[0][1][0]
    print(bbmodel.predict(pd_cf))
    pd_cf.loc[0] = counterfactual[0][2][0]
    print(bbmodel.predict(pd_cf))
    print(explainer.get_surrogate_fidelity())
    # explainer.get_rules()


def test_barbe_categorical(n_perturbations=5000):
    def categorical_named(x):
        if x <= 1:
            return "A"
        elif x <= 3:
            return "B"
        return "C"

    training_data, _ = _get_data()
    # make a discrete value
    # boolean
    training_data[list(training_data)[0]] = training_data[list(training_data)[0]] < 1
    # string
    training_data[list(training_data)[1]] = training_data[list(training_data)[1]].apply(categorical_named)
    # float / int
    training_data[list(training_data)[2]] = np.ceil(training_data[list(training_data)[2]])

    print("Running test: BARBE Categorical")
    data_row = training_data.drop('class', axis=1).iloc[10]
    training_labels = training_data['class']
    training_data = training_data.drop('class', axis=1)
    # For overfit
    # data_row = training_data.iloc[5]
    # training_labels = training_data['class']
    start_time = datetime.now()
    # so the random forest in itself cannot take categorical values :(
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)
    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data, training_labels, verbose=True, input_sets_class=True)
    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(explanation)
    print(explainer.get_categories())
    print(bbmodel.feature_importances_)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)


def test_barbe_categorical_given_scale_categorical(n_perturbations=5000):
    def categorical_named(x):
        if x <= 1:
            return "A"
        elif x <= 3:
            return "B"
        return "C"

    training_data, _ = _get_data()
    # make a discrete value
    # boolean
    training_data[list(training_data)[0]] = training_data[list(training_data)[0]] < 1
    # string
    training_data[list(training_data)[1]] = training_data[list(training_data)[1]].apply(categorical_named)
    # float / int
    training_data[list(training_data)[2]] = np.ceil(training_data[list(training_data)[2]])

    print("Running test: BARBE Categorical")
    data_row = training_data.drop('class', axis=1).iloc[10]
    training_labels = training_data['class']
    training_data = training_data.drop('class', axis=1)
    # For overfit
    # data_row = training_data.iloc[5]
    # training_labels = training_data['class']
    start_time = datetime.now()
    # so the random forest in itself cannot take categorical values :(
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)
    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data=training_data, verbose=True, input_sets_class=True)
    explainer = BARBE(input_scale=explainer.get_perturber(feature='scale'),
                      input_categories=explainer.get_perturber(feature='categories'),
                      feature_names=list(training_data),
                      verbose=True, input_sets_class=True)

    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(explanation)
    print(explainer.get_categories())
    print(bbmodel.feature_importances_)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)


# run all tests or specific tests if this is the main function
if __name__ == "__main__":
    test_iris_dataset()
