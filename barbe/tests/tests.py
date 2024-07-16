"""
This code contains tests that ensure the BARBE package is working correctly.
"""
from barbe.utils.lime_interface import LimeWrapper
from datetime import datetime
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import barbe.tests.tests_config as tests_config
import random
from numpy.random import RandomState
from barbe.explainer import BARBE

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

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


def test_produce_barbe_perturbations(n_perturbations=5000):
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed

    training_data, _ = _get_data()
    data_row = training_data.drop('class', axis=1).iloc[0]

    print("Running test: BARBE Perturbation")
    start_time = datetime.now()
    bp = BarbePerturber(training_data.drop('class', axis=1))
    perturbed_data = bp.produce_perturbation(n_perturbations, data_row=data_row)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(perturbed_data)



def test_simple_numeric():
   pass


def test_simple_text():
    pass


def test_iris_dataset():
    # IAIN add traffic dataset to this work with trained classification being
    #  the day of the week?
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    data_row = iris_df.iloc[50]
    training_labels = iris.target
    training_data = iris_df

    print("Running test: BARBE iris Run")
    start_time = datetime.now()
    bbmodel = RandomForestClassifier()
    bbmodel.fit(training_data, training_labels)
    # IAIN do we need the class to be passed into the explainer? Probably not...
    explainer = BARBE(training_data, training_labels, verbose=True, input_sets_class=True)
    explanation = explainer.explain(data_row, bbmodel)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(explanation)
    print(bbmodel.feature_importances_)


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
    (July 11th) TODO: Start work on paper review (look into paper share by Osmar)
    (July 11th) TODO: Ensure that data format of the input row is correct for both types of data
    (July 11th) TODO: Setup data to rerun a few times until the classification is correct
    (NOTE/July 13th) - Encoder is modified, but support ends up inconsistent so we need a fix
    (NOTE/July 13th) - Check why fidelity can be one yet it does not provide any rules
    (July 15th) TODO: Return the exact rules that were used on the case (e.g. 2<'sepal width (cm)'<5)
    (COMPLETE/July 16th) TODO: Make your own perturber
        - Needed to scale back the standard deviation to improve the results
    '''
    print("Running test: BARBE Glass Run")
    start_time = datetime.now()
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
    # explainer.get_rules()


# run all tests or specific tests if this is the main function
if __name__ == "__main__":
    test_iris_dataset()
