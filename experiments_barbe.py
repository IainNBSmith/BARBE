RANDOM_SEED = 1

import os
import re
import gc
import sys
import random
import json
import logging
import numpy as np
from numpy.random import RandomState
from collections import Counter, OrderedDict
import itertools
from pprint import pprint
#import pathos
# from multiprocessing import Pool, TimeoutError
from importlib import reload  
#import ipykernel
import requests
# Alternative that works for both Python 2 and 3:
from requests.compat import urljoin
#from notebook.notebookapp import list_running_servers


random.seed(RANDOM_SEED)
# np.random.seed(seed=RANDOM_SEED)
const_random_state = RandomState(RANDOM_SEED)
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, jaccard_score, precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel('CRITICAL')

import experiments_config

#sys.path.append(experiments_config.SIGDIRECT_PATH)
# from associative_classifier import AssociativeClassifier

############ importing LIME   ############
# if experiments_config.LIME_PATH not in sys.path:
#     sys.path.append(experiments_config.LIME_PATH)

LIME_PATH = '.\\lime'
SIGDIRECT_PATH = '.\\sigdirect'
sys.path.insert(1, LIME_PATH)
sys.path.insert(2, SIGDIRECT_PATH)

print(sys.path)

import lime
import lime.lime_tabular
import lime.barbe as barbe
import anchor
import rbo



def _reload_libs():
    global lime
#     xlime = reload(xlime)
#     xlime.lime_tabular = reload(xlime.lime_tabular)
    lime = reload(lime)
    lime.lime_tabular = reload(lime.lime_tabular)

def get_notebook_name():
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    print(list(servers))
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if nn['kernel']['id'] == kernel_id:
                return nn['notebook']['path'].split('/')[-1]
    return "backup_file"
#print(get_notebook_name())

def get_train_df(filename, has_index, header_index=None):
    df = pd.read_csv(filename, sep=',', header=header_index, na_values='?')
    if has_index:
        df = df.drop(df.columns[0], axis=1)
    return df

def get_test_df(filename, has_index, header_index=None):
    df = pd.read_csv(filename, sep=',', header=header_index, na_values='?')
    if has_index:
        df = df.drop(df.columns[0], axis=1)
    return df

def preprocess_data(train_df, test_df, class_index, dataset_name):
    # removing class label, so we can call get_dummies on the rest
    train_df.rename(columns={list(train_df.columns)[class_index]:'class'}, inplace=True)
    print(train_df)
    print({list(train_df.columns)[class_index]:'class'})
    train_class = train_df['class']
    print(train_class)
    train_df.drop(columns=['class'], inplace=True)
    print(train_df)
    
    test_df.rename(columns={list(test_df.columns)[class_index]:'class'}, inplace=True)
    test_class = test_df['class']
    test_df.drop(columns=['class'], inplace=True)

    # process categorical data
    infer_types = []
    df = pd.concat([train_df, test_df])     #Now they no class column

    for column in df.columns:
        if df[column].dtype=='object':
            infer_types.append("{}_CAT".format(column))
        else:
            infer_types.append("{}_NUM".format(column))
    datasets_info_dict[dataset_name]['_NUM'] = sum(['_NUM' in x for x in infer_types])
    datasets_info_dict[dataset_name]['_CAT'] = sum(['_CAT' in x for x in infer_types])
#     print('INFER_TYPES:', infer_types)
    
    df = pd.get_dummies(df)
    train_df = df[:train_df.shape[0]]
    test_df  = df[train_df.shape[0]:]

    print(train_df)
    print(test_df)
    assert set(train_df.columns)==set(test_df.columns)

    # process numerical data (standardization is independent[?] for train/test splits)
    continuous_column_names = [x for x in list(train_df.columns) if not '_' in str(x)]
    print(continuous_column_names)
    for column in continuous_column_names:
        # standardazing the column
        scaler = StandardScaler()
        train_df[column] = scaler.fit_transform(train_df[column].to_numpy().reshape((-1,1)))
        test_df[column]  = scaler.transform(test_df[column].to_numpy().reshape((-1,1)))
        
        # set NaN to 0
        train_df[column].fillna(0., inplace=True)
        test_df[column].fillna(0., inplace=True)
        
    train_df['class'] = train_class
    test_df['class']  = test_class
    
    print(train_df)
    print(test_df)
    return train_df, test_df
    
def get_data(dataset_name, dataset_info):
    # input: name of the dataset, and a dictionary containing its info
    # process input arguments
    ## TA: If file_counts is 2, then train file and test filenames are separated. If 1, then 1 single fine
    file_counts  = 2 if dataset_info['SEPARATE_FILES'] else 1
    class_index  = dataset_info['CLASS_INDEX']
    has_index    = dataset_info['HAS_INDEX']
    header_index = dataset_info['HEADER_ROW_NUMBER']
    
    # set train/test filenames
    if file_counts==2:
        filenames = [os.path.join(DATA_ROOT_DIR, dataset_info['FOLDER_NAME'], dataset_info['TRAIN_FILENAME']), 
                     os.path.join(DATA_ROOT_DIR, dataset_info['FOLDER_NAME'], dataset_info['TEST_FILENAME'])]
    else:
        filenames = [os.path.join(DATA_ROOT_DIR, dataset_info['FOLDER_NAME'], dataset_info['COMBINED_FILENAME'])]


    print(filenames)
    #filenames = ['/cshome/alamanik/barbetest/dataset/glass.data']
    filenames = ['./dataset/glass.data'] #IAIN
    
    if file_counts==1:
        # load from file
        df = get_train_df(filenames[0], has_index, header_index)
        # shuffle
        df = df.sample(frac=1, random_state=const_random_state)

        train_size = int(df.shape[0] * TRAIN_RATIO)
        train_df = df.iloc[:train_size]
        test_df  = df.iloc[train_size:]
        
    else:
        # load from file
        train_df = get_train_df(filenames[0], has_index, header_index)
        test_df  = get_test_df(filenames[1], has_index, header_index)
        # shuffle
        train_df = train_df.sample(frac=1, random_state=const_random_state)
        test_df  = test_df.sample(frac=1, random_state=const_random_state)

    # some of the preprocessing (one hot encoding + missing values + standardisation)
    dataset_info['initial_features'] = train_df.shape[1] -1
    
    train_df, test_df = preprocess_data(train_df, test_df, class_index, dataset_name)
    
    dataset_info['features'] = train_df.shape[1] -1
    dataset_info['initial_train_size'] = train_df.shape[0]
    dataset_info['initial_test_size'] = test_df.shape[0]
    dataset_info['original_train_df'] = train_df.copy()
    dataset_info['original_test_df']  = test_df.copy()
    
    if shrink_train_size and train_df.shape[0]>desired_train_size:
        train_df = train_df[:desired_train_size]
    if shrink_test_size  and test_df.shape[0]>desired_test_size:
        test_df = test_df[:desired_test_size]
    dataset_info['train_size'] = train_df.shape[0]
    dataset_info['test_size'] = test_df.shape[0]
        
    return train_df, test_df


# # Interpretable Models

# In[9]:


def get_clf(train_df, clf_type, info=None):
    if clf_type.lower()=='dt':
        return get_clf_dt(train_df, info['max_explanation_size'])
    if clf_type.lower()=='lr':
        return get_clf_lr(train_df, info['max_explanation_size'])
    raise Exception('Wrong interpretable model, select from "lr", "dt"')

def get_features(classifier_type, clf, row, info_dict):
    if classifier_type=='sd':
        clf_features = get_features_sd(clf, row)
    elif classifier_type=='dt':
        clf_features = get_features_dt(clf, row)
    elif classifier_type=='lr':
        clf_features = get_features_lr(clf, row, 
                                       info_dict['max_features'], 
                                       info_dict['num_labels'], 
                                       info_dict['predicted_label_index'])
    else:
        print('Incorrect classifier type. terminating ...', classifier_type)
        raise Exception("Incorrect classifier type")
    return clf_features


# ## 1. Decision Tree

# In[10]:


def get_clf_dt(train_df, max_depth):
    clf = sklearn.tree.DecisionTreeClassifier(random_state=const_random_state, max_depth=max_depth)
    clf.fit(train_df.drop('class', axis=1), train_df['class'])
    return clf

def test_classifier_dt(clf, test_df):
    predictions = clf.predict(test_df.drop('class', axis=1))    
    acc = sklearn.metrics.accuracy_score(test_df['class'], predictions)
    return acc

def get_features_dt(clf, row):
    feature = clf.tree_.feature
    print('Inside: get_features_dt, feature = ', feature)
    leave_id = clf.apply(row.values.reshape(1, -1))     #row.values is 1d array. row.values.reshape(1, -1) is 2d array
    #print(leave_id)
    node_indicator = clf.decision_path([row])
    #print(node_indicator)
    features = OrderedDict() # using as an ordered set
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
#     node_index = node_indicator.indices[:] #node_indicator.indptr[0]:node_indicator.indptr[1]]
    #print(node_index)

    for node_id in node_index:
        if leave_id[0] == node_id:  # <-- changed != to ==
            continue # <-- comment out
        else: # < -- added else to iterate through decision nodes
#             features.append(feature[node_id]+1)
            if feature[node_id]+1 not in features:
                features[feature[node_id]+1] = None
#             else:
#                 print('redundant!!!')
    #print(features)
    final_features = [*features]
    print('final_features = ', final_features)
#     print(final_features)
    return final_features

def get_dt_avg_explanation(dt_clf, train_df):
    # find the depth for each training instance, and then return the average among them
#     decision_paths = dt_clf.decision_path(train_df).toarray()
#     uniques = np.unique(decision_paths, axis=1)
#     print(decision_paths[:5])
#     print(np.count_nonzero(decision_paths, axis=1).max()-1, np.count_nonzero(decision_paths, axis=1).min()-1)
#     return np.count_nonzero(decision_paths, axis=1).mean() - 1 # they all have an extra node which is the leaf
    lens = 0
    print('*** ', dt_clf)
    print('*** ', train_df)
    for _,row in train_df.iterrows():
        exp = get_features_dt(dt_clf, row)
        lens += len(exp)
    return lens/train_df.shape[0]


# ## 2. Logistic Regression

# In[11]:


def get_clf_lr(train_df, max_features):
    try_cs1 = np.arange(1.,0,-.1)
    try_cs2 = np.arange(.1,0,-.01)
    try_cs3 = np.arange(.01,0,-.001)
    
    done = False
    for c in try_cs1:
        temp_clf = sklearn.linear_model.LogisticRegression(random_state=const_random_state, penalty='l1', fit_intercept=True, C=c, n_jobs=-1, solver='liblinear')
        temp_clf.fit(train_df.drop('class', axis=1), train_df['class'])
        lengths = [len(x.nonzero()[0]) for x in temp_clf.coef_]
        if np.max(lengths) <= max_features:
            done = True
            break
    if done:
        return temp_clf
    for c in try_cs2:
        temp_clf = sklearn.linear_model.LogisticRegression(random_state=const_random_state, penalty='l1', fit_intercept=True, C=c, n_jobs=-1, solver='liblinear')
        temp_clf.fit(train_df.drop('class', axis=1), train_df['class'])
        lengths = [len(x.nonzero()[0]) for x in temp_clf.coef_]
        if np.max(lengths) <= max_features:
            done = True
            break
    if done:
        return temp_clf
    for c in try_cs3:
        temp_clf = sklearn.linear_model.LogisticRegression(random_state=const_random_state, penalty='l1', fit_intercept=True, C=c, n_jobs=-1, solver='liblinear')
        temp_clf.fit(train_df.drop('class', axis=1), train_df['class'])
        lengths = [len(x.nonzero()[0]) for x in temp_clf.coef_]
        if np.max(lengths) <= max_features:
            done = True
            break
#     print('c:', c)
    return temp_clf
    
def test_classifier_lr(clf, test_df):
    predictions = clf.predict(test_df.drop('class', axis=1))    
    acc = sklearn.metrics.accuracy_score(test_df['class'], predictions)
    return acc

def get_features_lr(clf, row, max_features, num_classes, label_index):
    if num_classes<=2:
        idx = 0
    else:
        idx = label_index
    all_params = clf.coef_
    return set(np.where(all_params[idx]!=0.)[0]+1)
#     return set(np.argsort(all_params[idx])[:].tolist())

def get_lr_avg_explanation(clf, train_df):
    return max([len(x.nonzero()[0]) for x in clf.coef_])


# # Global variables

# In[12]:


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

datasets_info_dict = experiments_config.all_datasets_info_dict.copy()

datasets_info_dict = {
                    'glass':experiments_config.all_datasets_info_dict['glass'],
#                      'wine':experiments_config.all_datasets_info_dict['wine'],
#                      'hungarian':experiments_config.all_datasets_info_dict['hungarian'],
#                      'hepatitis':experiments_config.all_datasets_info_dict['hepatitis'],
#                      'poker':experiments_config.all_datasets_info_dict['poker'],
                     }


info = {'max_explanation_size':5}

remove_datasets = [
    'online_shoppers_intention', # slow with 1k!
    'breast-cancer', # very slow even in 200!, but good results!
    'car', # has a small tree! f1=55, fidel:82
    'nursery', # precision=1, recall=.5
    'adult', # slow, 1k won't finish in 2h
]
# remove_datasets = ['online_shoppers_intention']

for x in remove_datasets:
    if x in  datasets_info_dict:
        del datasets_info_dict[x]

print(datasets_info_dict.items())
for dataset, dataset_info in datasets_info_dict.items():
    print('---', dataset)
    print('---', dataset_info)
    train_df, test_df = get_data(dataset, dataset_info)
    dataset_info['train_df'] = train_df
    dataset_info['test_df'] = test_df
    dataset_info['nclasses'] = train_df['class'].unique().shape[0]


for dataset, dataset_info in datasets_info_dict.items():
    ############ Decision Tree ############
    print('111')
    if 'dt_clf' not in dataset_info:
        print('dt_clf')
        dt_clf = get_clf(dataset_info['original_train_df'], 'dt', info)
        avg_explanation_len = get_dt_avg_explanation(dt_clf, dataset_info['train_df'].drop(columns=['class']))
        print('avg_explanation_len = ', avg_explanation_len)
        dataset_info['dt_len'] = avg_explanation_len
        dataset_info['dt_acc'] = test_classifier_dt(dt_clf, dataset_info['test_df'])
        dataset_info['dt_clf'] = dt_clf
    ############ Logistic Regression ############
    if 'lr_clf' not in dataset_info:
        print('lr_clf')
        lr_clf = get_clf(dataset_info['original_train_df'], 'lr', info)
        avg_explanation_len = get_lr_avg_explanation(lr_clf, dataset_info['train_df'].drop(columns=['class']))
        dataset_info['lr_len'] = avg_explanation_len
        dataset_info['lr_acc'] = test_classifier_lr(lr_clf, dataset_info['test_df'])
        dataset_info['lr_clf'] = lr_clf
    print('222')

def get_nonzero_features_lime(row, clf, num_features, explainer, num_labels, 
                              max_valid_features, predicted_label_index, num_samples, xlime_mode, seed):
    """ For a given class, return top k features"""
    random.seed(seed)
    np.random.seed(seed)
    classes = list(range(num_labels))
    print('** classes ', classes, 'and row ', row)
    print('Caller 2: row.to_numpy() = ', row.to_numpy())
    x = explainer.explain_instance(row.to_numpy(), 
                               clf.predict_proba, 
                               num_features=max_valid_features, 
                               labels=[predicted_label_index],
                               num_samples=num_samples)
#         print(x.local_pred)
#         fidelity = 1.0
    if x.local_pred>=1.0/num_labels:
        fidelity = 1.0
    else:
        fidelity = 0.0
    feature_score_pairs = x.as_map()[predicted_label_index]
#     print('feature_score_pairs.shape', len(feature_score_pairs))
    feature_score_pairs = [(x[0]+1, x[1]) for x in feature_score_pairs if x[1]!=0.]
    feature_score_pairs = sorted(feature_score_pairs, key=lambda x:x[1], reverse=True)
    nonzero_features = [x[0] for x in feature_score_pairs[:max_valid_features]]
    print('nonzero_features = ', nonzero_features)
    if fidelity==0.0:##
        nonzero_features = []
    return nonzero_features, fidelity

def get_nonzero_features_anchor(row, clf, num_features, explainer, 
                              max_valid_features, predicted_label_index, num_samples, seed):
    """ For a given class, return top k features"""
    random.seed(seed)
    np.random.seed(seed)
    
#         def explain_instance(self, data_row, classifier_fn, threshold=0.95,
#                           delta=0.1, tau=0.15, batch_size=100,
#                           max_anchor_size=None,
#                           desired_label=None,
#                           beam_size=4, **kwargs):

#     print('shape:', row.to_numpy().shape)
    x = explainer.explain_instance(row.to_numpy().reshape(1,-1), 
                                   clf.predict, 
                                   max_anchor_size=max_valid_features, 
                                   desired_label=predicted_label_index,
#                                    num_samples=num_samples,
                                   coverage_samples=num_samples,
                                  )
#     print(x.names(), x.features())
    features = x.features()#[predicted_label_index]
    features = [x+1 for x in features]
#     feature_score_pairs = [(x[0]+1, x[1]) for x in feature_score_pairs if x[1]!=0.]
#     nonzero_features = [x[0] for x in feature_score_pairs]
    return features

def get_nonzero_features_shap(row, clf, num_features, explainer, 
                              max_valid_features, predicted_label_index, num_samples, seed):
    """ For a given class, return top k features"""
    random.seed(seed)
    np.random.seed(seed)
    x = explainer.shap_values(row, nsamples=num_samples)
    features_scores = x[predicted_label_index]
#     print(features_scores)
    sorted_features = sorted(list(enumerate(features_scores)), key=lambda x:abs(x[1]), reverse=True)
    ret = [x[0]+1 for x in sorted_features[:max_valid_features]]
    return ret


# In[18]:

##Will be called for each row in test df
def evaluate_instance(args):    
    ((idx, row), clf, classifier_type, num_features, num_labels, 
                      explainer, num_samples, xlime_mode, seed, ordered_class_labels, method, max_features) = args
    print('Evaluating: ', ((idx, row), clf, classifier_type, num_features, num_labels, 
                      explainer, num_samples, xlime_mode, seed, ordered_class_labels, method, max_features))

    random.seed(seed)
    np.random.seed(seed)
#     print(idx)
    fidelity = 1
    predicted_label_index = ordered_class_labels.index(clf.predict(row.values.reshape(1, -1))[0])
    print(row.values, row.values.reshape(1, -1), clf.predict(row.values.reshape(1, -1))[0], predicted_label_index, ordered_class_labels)
    clf_features = get_features(classifier_type, clf, row, {'max_features':max_features, 'num_labels':num_labels, 'predicted_label_index':predicted_label_index})
    print(clf_features)
    if method=="XLIME":
#         my_modes = [["FOURTEEN", "SIXTEEN"], ["SEVENTEEN", "EIGHTEEN"], "FIFTEEN" ]
#         my_modes = [["FOURTEEN", "SEVENTEEN"]] 
        my_modes = ["FOURTEEN", "SEVENTEEN", "SIXTEEN", "EIGHTEEN"]
        explainer_features = []
        temp_seed = seed
        for i in range(REPEAT_COUNT):
            b, fidelity = get_nonzero_features_lime(row, clf, num_features, 
                                                  explainer, num_labels, max_features,                                                      
                                                  predicted_label_index, num_samples, 
#                                                   ["FOURTEEN","FIFTEEN"],
                                                  my_modes[i] , # "FOURTEEN"
                                                  seed, # temp_seed, # seed, # 
                                                   )
            temp_seed += 11
            if fidelity==1.0:
#                 if i>0:
#                     print(i)
                if len(explainer_features)>0:
                    explainer_features.extend([x for x in b if x not in set(explainer_features)])
                else:
                    explainer_features = b
                break
        
        explainer_features = explainer_features[:max_features]
    elif method=="LIME":
        print('----')
        explainer_features, fidelity = get_nonzero_features_lime(row, clf, num_features, 
                                                  explainer, num_labels, max_features,                                                      
                                                  predicted_label_index, num_samples, xlime_mode,
                                                  seed)

    elif method=="ANCHOR":
        explainer_features = set(get_nonzero_features_anchor(row, clf, num_features, 
                                                  explainer, max_features, 
                                                  predicted_label_index, num_samples,
                                                  seed))
    elif method=="SHAP":
        explainer_features = set(get_nonzero_features_shap(row, clf, num_features, 
                                                  explainer, max_features, 
                                                  predicted_label_index, num_samples,
                                                  seed))
    else:
        raise Exception("Incorrect method selected", method)
#     print(idx, 
#           fidelity,
#           list(clf_features), 
#           list(explainer_features),
#           sum([x in clf_features for x in explainer_features])/len(explainer_features) if len(explainer_features)>0 else 0.0,
#           sum([x in explainer_features for x in clf_features])/len(clf_features) if len(clf_features)>0 else 0.0)
    fout.write("{}, {}, {}, {}\n".format(idx, 
                                         list(clf_features), 
                                         list(explainer_features), 
                                         sum([x in clf_features for x in explainer_features])/len(explainer_features) if len(explainer_features)>0 else 0.0,
                                         sum([x in explainer_features for x in clf_features])/len(clf_features) if len(clf_features)>0 else 0.0))
    return list(clf_features), list(explainer_features), fidelity
    

def evaluate_explanations_parallel(dataset_name, clf, train_df, test_df, classifier_type, num_samples, around_instance, seed, max_features, method='LIME', xlime_mode='ONE'):
    print('Function evaluate_explanations_parallel with params = ', dataset_name, clf, classifier_type, num_samples, around_instance, seed, max_features, method, xlime_mode)
    random.seed(1)
    np.random.seed(1)
    train_df2 = train_df.drop('class', axis=1)   #train_df2 and test_df2 is without class labels
    test_df2 = test_df.drop('class', axis=1)
    ordered_class_labels = sorted(list(set(train_df['class'].values)))
    print('train_df ordered_class_labels: ', ordered_class_labels)
    columns = list(train_df2.columns)
    categorical_features        = [x for x in columns if '_' in str(x)]
    categorical_feature_indices = [columns.index(x) for x in columns if '_' in str(x)]
    categorical_features_map    = {columns.index(x):x for x in columns if '_' in str(x)}
    all_features = train_df2.columns.values
    print('dataset:', dataset_name, 'method:', method, 'seed:', seed, 'num_samples:', num_samples, 'test size:', test_df2.shape)
    print('all_features')
    print(all_features)
    fout.write('dataset: {} method: {} seed: {} num_sampes: {} test size: {}\n'.format(dataset_name, method, seed, num_samples, test_df2.shape))
    
    if method=='XLIME':
#         discretizers = ['decile', 'eightile', 'sixile', 'quartile'] 
#         for i in range(4):
#             try:
        explainer = barbe.BarbeExplainer(train_df2.values,
                                         categorical_features=categorical_feature_indices,
                                         feature_names=all_features,
                                         verbose=False,
                                         class_names=ordered_class_labels,
                                         mode='classification',
                                         random_state=RandomState(seed),
                                         discretizer='decile',
#                                        discretizer=discretizers[i],
    #                                    training_labels=train_df['class'].values,
                                         feature_selection='none'
                                         )
    
    elif method=='LIME':
        print('My method is LIME')
        discretizers = ['decile', 'eightile', 'sixile', 'quartile'] 
        for i in range(1):
            try:
                # IAIN replaced lime.lime_tabular with lime.barbe ...
                explainer = lime.lime_tabular.LimeTabularExplainer(train_df2.values,  
                                                       categorical_features=categorical_feature_indices, 
                                                       feature_names=all_features,
                                                       verbose=False, 
                                                       class_names=ordered_class_labels,
                                                       mode='classification', 
                                                       sample_around_instance=around_instance, 
                                                       random_state=RandomState(seed),
                                                       discretizer=discretizers[i],
                                                       #training_labels=train_df['class'].values,
                                                      )
                print(explainer)
                break
            except Exception as e:
                print(str(e))
                pass
    elif method=='ANCHOR':
#          def __init__(self, class_names, feature_names, data=None,
#                  categorical_names=None, ordinal_features=[]):
#         explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, 
#                                                           dataset.data, dataset.categorical_names)
        explainer = anchor.anchor_tabular.AnchorTabularExplainer(ordered_class_labels,
                                                       train_data=train_df2.values,  
                                                       categorical_names=categorical_features_map, 
                                                       feature_names=all_features,
#                                                        verbose=False, 
#                                                        class_names=ordered_class_labels, # error
#                                                        mode='classification',  # error
#                                                        sample_around_instance=around_instance,  # error
#                                                        random_state=RandomState(seed), # error
                                                       discretizer='decile'
                                                      )

    elif method=='SHAP':
        explainer = shap.KernelExplainer(clf.predict_proba, train_df2.values) #, link=<shap.common.IdentityLink object>, **kwargs)
    else:
        raise Exception("incorrect explanation method provided to evaluate_explanations:", method)
        
    num_labels = len(ordered_class_labels) # 0#max(train_df['class']) - train_df.shape[1]
    
    # modify this based on the type of the experiments (tuning --> train_df2)/(evaluating --> test_df2)
#     chosen_dataset = train_df2
    chosen_dataset = test_df2
    print(chosen_dataset.head())

    if method in ('XLIME', 'ANCHOR'):
        print('1', method)
        with pathos.multiprocessing.ProcessPool(ncpus=PROCESS_COUNT) as pool:
            ret = pool.map(evaluate_instance, zip(
                                                  chosen_dataset.iterrows(),
                                                  itertools.cycle([clf]),
                                                  itertools.cycle([classifier_type]),
                                                  itertools.cycle([chosen_dataset.shape[1]]),
                                                  itertools.cycle([num_labels]),
                                                  itertools.cycle([explainer]),
                                                  itertools.cycle([num_samples]),
                                                  itertools.cycle([xlime_mode]),
                                                  itertools.cycle([seed]),
                                                  itertools.cycle([ordered_class_labels]),
                                                  itertools.cycle([method]),
                                                  itertools.cycle([max_features])
                                                 ),                                                                          
                                             )
    else:
        # print('2', method, ordered_class_labels, list(zip(
        #                                           chosen_dataset.iterrows(),
        #                                           itertools.cycle([clf]),
        #                                           itertools.cycle([classifier_type]),
        #                                           itertools.cycle([chosen_dataset.shape[1]]),
        #                                           itertools.cycle([num_labels]),
        #                                           itertools.cycle([explainer]),
        #                                           itertools.cycle([num_samples]),
        #                                           itertools.cycle([xlime_mode]),
        #                                           itertools.cycle([seed]),
        #                                           itertools.cycle([ordered_class_labels]),
        #                                           itertools.cycle([method]),
        #                                           itertools.cycle([max_features])
        #                                          )))
        
        ret = map(evaluate_instance, zip(
                                                  chosen_dataset.iterrows(),
                                                  itertools.cycle([clf]),
                                                  itertools.cycle([classifier_type]),
                                                  itertools.cycle([chosen_dataset.shape[1]]),
                                                  itertools.cycle([num_labels]),
                                                  itertools.cycle([explainer]),
                                                  itertools.cycle([num_samples]),
                                                  itertools.cycle([xlime_mode]),
                                                  itertools.cycle([seed]),
                                                  itertools.cycle([ordered_class_labels]),
                                                  itertools.cycle([method]),
                                                  itertools.cycle([max_features])
                                                 ),                                                                          
                                             )

        
    all_clf_features, all_exp_features, fidelities = zip(*ret)
    print('--- Returned ', all_clf_features, all_exp_features, fidelities)
    exit()
    return all_clf_features, all_exp_features, fidelities


# In[19]:


# Evaluation functions
import rbo
def analyze_outputs(all_features, initial_all_clf_features, initial_all_exp_features, fidelities):
    random.seed(1)
    np.random.seed(1)
    if fidelity_division:
        all_clf_features = [x for x,fidel in zip(initial_all_clf_features, fidelities) if fidel]
        all_exp_features = [x for x,fidel in zip(initial_all_exp_features, fidelities) if fidel]
    else:
        all_clf_features = initial_all_clf_features
        all_exp_features = initial_all_exp_features
        
    print(len(all_clf_features), sum(fidelities))
#     print('shapes:', len(all_clf_features), len(all_exp_features))
    mlb = MultiLabelBinarizer()
    mlb.fit([[i+1] for i,x in enumerate(all_features)])
#     print('unique 1:', set(itertools.chain(*all_clf_features)))
#     print('unique 2:', set(itertools.chain(*all_exp_features)))
    print(classification_report(mlb.transform(all_clf_features), mlb.transform(all_exp_features), output_dict=True)['samples avg'])
    fout.write(str(classification_report(mlb.transform(all_clf_features), mlb.transform(all_exp_features), output_dict=True)['samples avg'])+'\n')

    print('f0.5-score:', precision_recall_fscore_support(mlb.transform(all_clf_features), mlb.transform(all_exp_features), beta=0.5, average='samples')[2])
    fout.write('f0.5-score: ' + str(precision_recall_fscore_support(mlb.transform(all_clf_features), mlb.transform(all_exp_features), beta=0.5, average='samples')[2])+'\n')

    print('Jaccard similarity:', jaccard_score(mlb.transform(all_clf_features), mlb.transform(all_exp_features), 
                                               average='samples'))
    fout.write('Jaccard similarity: ' + str(jaccard_score(mlb.transform(all_clf_features), mlb.transform(all_exp_features), 
                                               average='samples'))+'\n')
    # TODO 1: add RBO to this method
    # TODO 2: Are SHAP, Anchor outputs ordered?
    rbo_scores = []
    rbo_scores2 = []
    for clf_features, exp_features in zip(all_clf_features, all_exp_features):
#         rbo_score = rbo.RankingSimilarity(clf_features, exp_features).rbo()
#         rbo_scores.append(rbo_score)
#         print('RBO: {} {} {}'.format(rbo_score, clf_features, exp_features))
        if len(clf_features)>0 and len(exp_features)>0:
            rbo_score2 = rbo(clf_features, exp_features, p=0.5)
#             rbo_scores2.append(rbo_score2.min + rbo_score2.res/2)
            rbo_scores2.append(rbo_score2.ext)
        else:
            rbo_score2 = 0.
            rbo_scores2.append(0.)
#         print('RBO: {}'.format(rbo_score2))
#     print('RBO AVG: {}'.format(sum(rbo_scores)/len(rbo_scores)))
    print('RBO2 AVG: {}'.format(sum(rbo_scores2)/len(rbo_scores2)))
    fout.write('RBO2 AVG: {}\n'.format(sum(rbo_scores2)/len(rbo_scores2)))
    print('FIDELITY AVG: {}'.format(sum(fidelities)/len(fidelities)))
    fout.write('FIDELITY AVG: {}\n'.format(sum(fidelities)/len(fidelities)))


# ## Experiments

# In[20]:



############### experiment started ###################
theNotebook = 'final_anchor' # get_notebook_name().split('.')[0]
fout = open('{}.log'.format(theNotebook), 'a', 1)
samples_range = range(1000, 6000, 1000)
print(samples_range)
# samples_range = range(100, 1100, 100)
NUM_ITERATIONS = 5
classifier_type = 'dt'
around_instance = True
_reload_libs()
xlime_mode = ["FOURTEEN"]
iteration_seed = 0

for i in range(NUM_ITERATIONS):
    iteration_seed += 1
    for dataset, dataset_info in datasets_info_dict.items():
        print('Barbe experiment started with iterations = ', i, ' Dataset: ', dataset)
        seed = iteration_seed
        gc.collect()
        
        try:
            clf = dataset_info['{}_clf'.format(classifier_type)]
            print('*** Classifier = ', clf)
            train_df = dataset_info['train_df']    #Train_df contains class label
            test_df  = dataset_info['test_df']     #Test_df contains class label
            all_features = train_df.drop('class', axis=1).columns.values
            
            ## evaluate LIME
            for num_samples in samples_range:
                ouputs_clf, outputs_exp, fidelities = evaluate_explanations_parallel(dataset, clf, train_df, test_df, classifier_type, num_samples, around_instance, seed, info['max_explanation_size'], 'LIME')
                #analyze_outputs(all_features, ouputs_clf, outputs_exp, fidelities)

        except Exception as e:
            print('EXCEPTION:', str(e))
            fout.write('EXCEPTION' + str(e)+'\n')
            raise e

