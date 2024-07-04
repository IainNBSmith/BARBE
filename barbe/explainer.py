"""
This file contains the explainer part of BARBE and will call to all other parts of BARBE that make
 sense.
"""

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

from copy import deepcopy

# Import wrappers for other packages
from utils.sigdirect_interface import SigDirectWrapper
from utils.lime_interface import LimeWrapper


class BARBE:
    '''
    Example of a call
    Clearly way too much is being handled by something else, it should be simple...


    def evaluate_explanations_parallel(dataset_name, clf, train_df, test_df, classifier_type, num_samples, around_instance, seed, max_features, method='BARBE', xlime_mode='ONE'):

    # IAIN these steps should be built in? Or is there something I am missing...?
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



    explainer = barbe.BarbeExplainer(train_df2.values,
                                       categorical_features=categorical_feature_indices,
                                       feature_names=all_features,
                                       verbose=False,
                                       class_names=ordered_class_labels,
                                       mode='classification',
                                       sample_around_instance=around_instance,
                                       random_state=RandomState(seed),
                                       discretizer=discretizers[i]
                                       )
                print(explainer)
    '''
    def __init__(self, verbose=False, mode='tabular', n_perturbations=5000):
        # IAIN include model settings, make the model here?
        # IAIN do we include the perturbation type here? Is it consistent?
        # IAIN eventually method will include 'text' too.
        # IAIN raise warning when perturbations are not the default and text is called

        # IAIN should include some checks in here for mode and perturbation validity

        self._verbose = verbose
        self._n_perturbations = n_perturbations
        self._mode = mode
        self._perturbed_data = None
        self._blackbox_classification = {'input': None, 'perturbed': None}
        self._surrogate_classification = {'input': None, 'perturbed': None}
        self._surrogate_model = None  # SigDirect model trained on the new points
        self._explanation = "No explanations done yet."

    def __str__(self):
        return self._explanation

    def _fit_surrogate_model(self, input_data, input_model):
        # IAIN tabular and text perturbations differ in an important way
        if self._mode in 'tabular':
            self._generate_perturbed_tabular(input_data)
        elif self._mode in 'text':
            # IAIN should currently pass an error
            pass
        # get black box predictions
        self._blackbox_classification['input'] = input_model.predict(input_data)
        self._blackbox_classification['perturbed'] = input_model.predict(self._perturbed_data)

        # fit the surrogate through the wrapper
        self._surrogate_model = SigDirectWrapper.fit(self._perturbed_data,
                                                     self._blackbox_classification['perturbed'])

        self._surrogate_classification['input'] = self._surrogate_model.predict(input_data)
        self._surrogate_classification['perturbed'] = self._surrogate_model.predict(self._perturbed_data)

    def _check_input_data(self, input_data):
        # IAIN checks if input is valid (tabular data is tabular, text data is text)
        #  and put it into a common format (pandas dataframe or string)
        if self._mode in 'tabular':
            pass
        elif self._mode in 'text':
            # IAIN should currently pass an error
            pass
        pass

    def _check_input_model(self, input_model):
        # IAIN ensures that the input model has a predict function and output is as expected
        #  or can be formatted into a list (,k)
        if self._mode in 'tabular':
            pass
        elif self._mode in 'text':
            # IAIN should currently pass an error
            pass
        pass

    def _generate_perturbed_tabular(self, input_data):
        # calls to set method to perturb the data, based on the mode
        # IAIN should check which columns are categorical and which are numeric
        self._perturbed_data = None
        pass

    def _generate_perturbed_text(self, input_data):
        # calls to set method to perturb the data, based on the mode
        # IAIN pass a warning / error
        self._perturbed_data = None
        pass

    def get_surrogate_model(self):
        return deepcopy(self._surrogate_model)

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score):
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        pass

    def get_rules(self):
        # IAIN this will output rules and their translations as learned by the model
        pass

    def get_features(self):
        # IAIN same as previous named get features
        pass

    def explain(self, input_data, input_model, fit_new_surrogate=True):
        # IAIN should include the surrogate similarity to original model
        # IAIN follow how it is called at the moment
        # IAIN the way it is called by them kinda sucks
        # IAIN in the case of not fitting a new surrogate use the previous one to explain a new
        #  sample (results may vary hence why it is always true by default)

        # check input before going further
        self._check_input_model(input_model)
        self._check_input_data(input_data)

        # cannot test without a new surrogate
        if self._surrogate_model is None:
            # IAIN push a warning later
            fit_new_surrogate = True

        if fit_new_surrogate:
            # fit the SigDirect model
            self._fit_surrogate_model(input_data, input_model)

        # IAIN need to add something here that takes the input data for the explanation

        # IAIN expecting a fit model explain the result for the new data
        self._explanation = (self.get_features() + " \n " +
                             self.get_surrogate_fidelity() + " \n " +
                             self.get_rules())

        return self._explanation
