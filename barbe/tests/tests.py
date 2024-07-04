"""
This code contains tests that ensure the BARBE package is working correctly.
"""
from barbe.utils.lime_interface import LimeWrapper
from datetime import datetime
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tests_config


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
DATA_ROOT_DIR = "./dataset"

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
    df = pd.read_csv(filename, sep=',', header=header_index, na_values='?')
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
        test_df[column] = scaler.transform(test_df[column].to_numpy().reshape((-1, 1)))

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
    # dataset_name, dataset_info
    filenames = ['./dataset/glass.data']  # IAIN

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
    dataset_info['initial_test_size'] = test_df.shape[0]
    dataset_info['original_train_df'] = train_df.copy()
    dataset_info['original_test_df'] = test_df.copy()

    if shrink_train_size and train_df.shape[0] > desired_train_size:
        train_df = train_df[:desired_train_size]
    if shrink_test_size and test_df.shape[0] > desired_test_size:
        test_df = test_df[:desired_test_size]
    dataset_info['train_size'] = train_df.shape[0]
    dataset_info['test_size'] = test_df.shape[0]

    return train_df, test_df


def test_produce_lime_perturbations(n_perturbations=5000):
    training_data, _ = _get_data()

    print("Running test: Lime Perturbation")
    start_time = datetime.now()
    lw = LimeWrapper(training_data)
    perturbed_data = lw.produce_perturbation(data_row, n_perturbations)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(perturbed_data)


def test_simple_numeric():
    pass


def test_simple_text():
    pass


def test_glass_dataset():
    pass


# run all tests or specific tests if this is the main function
if __name__ == "__main__":
    test_produce_lime_perturbations()
