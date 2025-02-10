import time

import pandas as pd
from sklearn.neural_network import MLPClassifier  # neural network to train
import dice_ml
import matplotlib.pyplot as plt
from barbe.explainer import BARBE
from sklearn.decomposition import PCA
import tensorflow_datasets as tfds
import pickle
import numpy as np


def import_mnist(n_train=5000, n_test=10):
    #splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
    #df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    #df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])
    ds_train, ds_test = tfds.load("mnist",
                                  split=['train', 'test'],
                                  shuffle_files=False,
                                  as_supervised=True,
                                  with_info=False)
    ds = ds_train.take(n_train)
    df_train, train_label = list(), list()
    for temp_sample, temp_label in tfds.as_numpy(ds):
        df_train.append(temp_sample.flatten())
        train_label.append(temp_label)

    ds = ds_test.take(n_test)
    df_test, test_label = list(), list()
    for temp_sample, temp_label in tfds.as_numpy(ds):
        df_test.append(temp_sample.flatten())
        test_label.append(temp_label)
    return np.array(df_train, dtype=float), train_label, np.array(df_test, dtype=float), test_label


def import_mnist_pca(n_train=5000, n_test=10, pca=None):
    df_train, label_train, df_test, label_test = import_mnist(n_train=n_train, n_test=n_test)
    if pca is None:
        encoder = PCA(n_components=30)
        encoder.fit(df_train)
    else:
        encoder = pca
    return encoder.transform(df_train), label_train, encoder.transform(df_test), label_test, encoder


def display_mnist(data_point, ax):
    # display a single row of mnist data on the given axes
    #fig = plt.figure
    ax.imshow(data_point.reshape((28, 28)), cmap='gray')
    #plt.show()


def test_train_save_bbmodel():
    # train and save a bbmodel to predict mnist numbers
    train_dataset, train_label, test_dataset, test_label = import_mnist(n_train=5000)
    clf = MLPClassifier()
    clf.fit(train_dataset, train_label)
    print(clf.predict(test_dataset))
    print(test_label)
    with open("../pretrained/mnist_nn.pickle", "wb") as f:
        pickle.dump(clf, f)


def test_train_save_bbmodel_pca():
    # train and save a bbmodel to predict mnist numbers
    train_dataset, train_label, test_dataset, test_label, pca = import_mnist_pca(n_train=5000)
    clf = MLPClassifier()
    clf.fit(train_dataset, train_label)
    print(clf.predict(test_dataset))
    print(test_label)
    with open("../pretrained/mnist_nn_pca.pickle", "wb") as f:
        pickle.dump((clf, pca), f)


def open_pretrained_bbmodel(pca=True):
    # open a pretrained model saved by train_save_bbmodel()
    if not pca:
        with open("../pretrained/mnist_nn.pickle", "rb") as f:
            model = pickle.load(f)
        return model, None
    else:
        with open("../pretrained/mnist_nn_pca.pickle", "rb") as f:
            model, pca = pickle.load(f)
        return model, pca


def test_dice():
    # create axis
    ax = None  # use  matplot lib to make multiple plots to display of all counterfactuals
    # test and display DiCE output on numbers
    # get training label from pre-trained neural network
    bbmodel, pca = open_pretrained_bbmodel(pca=False)
    fig = plt.figure(figsize=(10,8))

    train_dataset, train_label, test_dataset, test_label = import_mnist(n_train=5000)
    print(train_dataset.shape)
    train_df = pd.DataFrame(data=train_dataset)
    train_df['label'] = train_label
    # Dataset for training an ML model
    features = list(train_df)
    features.pop()
    print(train_df)
    d = dice_ml.Data(dataframe=train_df,  # all features are continuous
                     continuous_features=features,
                     outcome_name='label')

    m = dice_ml.Model(model=bbmodel,
                      backend='sklearn')#, func="ohe-min-max")

    # DiCE explanation instance
    exp = dice_ml.Dice(d, m, method='random')  # gradient if using pytorch or tensorflow models

    test_df = pd.DataFrame(data=test_dataset)
    # ind2 = 4 (swap to 9) ind8 = 3 (swap to 8) ind4 = 7 (swap to 1)
    ind_counter = [9, 8, 1]
    ind_use = [2, 8, 4]
    j = 1
    for i in range(len(ind_use)):

        ind_example = ind_use[i]
        counter = ind_counter[i]
        ax = fig.add_subplot(len(ind_counter), 5, j)
        j+=1
        display_mnist(test_df[ind_example:(ind_example + 1)].to_numpy(), ax)
        dice_exp = exp.generate_counterfactuals(test_df[ind_example:(ind_example+1)],
                                                total_CFs=4,
                                                desired_class=counter,
                                                proximity_weight=1,
                                                diversity_weight=0.5)
        print(dice_exp.cf_examples_list[0].final_cfs_df)
        print(dice_exp.cf_examples_list[0].final_cfs_df.shape)
        if pca is not None:
            display_mnist(pca.inverse_transform(test_df[ind_example:(ind_example+1)].to_numpy()), ax)
            display_mnist(pca.inverse_transform(dice_exp.cf_examples_list[0].final_cfs_df.drop(columns='label').iloc[0].to_numpy()), ax)
        else:
            ax = fig.add_subplot(len(ind_counter), 5, j)
            display_mnist(dice_exp.cf_examples_list[0].final_cfs_df.drop(columns='label').iloc[0].to_numpy(), ax)
            j+=1
            ax = fig.add_subplot(len(ind_counter), 5, j)
            display_mnist(dice_exp.cf_examples_list[0].final_cfs_df.drop(columns='label').iloc[1].to_numpy(), ax)
            j+=1
            ax = fig.add_subplot(len(ind_counter), 5, j)
            display_mnist(dice_exp.cf_examples_list[0].final_cfs_df.drop(columns='label').iloc[2].to_numpy(), ax)
            ax = fig.add_subplot(len(ind_counter), 5, j)
            j+=1
            display_mnist(dice_exp.cf_examples_list[0].final_cfs_df.drop(columns='label').iloc[3].to_numpy(), ax)
            j+=1
        print("Original Class: ", bbmodel.predict(test_df[ind_example:(ind_example+1)]))
        print("Counterfactual Class: ", bbmodel.predict(dice_exp.cf_examples_list[0].final_cfs_df.drop(columns='label')))

    plt.show()


def test_barbe():
    # create axis
    ax = None  # use  matplot lib to make multiple plots to display of all counterfactuals
    bbmodel, pca = open_pretrained_bbmodel(pca=True)
    train_dataset, train_label, test_dataset, test_label, _ = import_mnist_pca(n_train=5000, pca=pca)
    print(train_dataset.shape)
    train_df = pd.DataFrame(data=train_dataset)
    # Dataset for training an ML model
    # TODO: generate the variations then simplify the features based on PCA?
    explainer = BARBE(training_data=train_df,
                      perturbation_type='normal',
                      dev_scaling_factor=1,
                      n_bins=5,
                      feature_names=list(train_df),
                      verbose=False,
                      input_sets_class=False,
                      n_perturbations=5000)
    test_df = pd.DataFrame(data=test_dataset)
    explanation = explainer.explain(test_df.iloc[2], bbmodel)
    print("Fidelity: ", explainer.get_surrogate_fidelity())
    # make sure to iloc at 2
    cf, _, _ = explainer.get_counterfactual_explanation(test_df.iloc[2],
                                                        wanted_class=9,
                                                        n_counterfactuals=1)
    display_mnist(pca.inverse_transform(cf[0]), ax)


#a, aa, b, bb = import_mnist(n_train=100, n_test=10)
#display_mnist(a[9], None)
