# IAIN provides plotting utilities to the visualizer
import numpy as np
import seaborn as sns
import matplotlib as mtp
from barbe.explainer import BARBE
from barbe.utils.bbmodel_interface import BlackBoxWrapper
from barbe.perturber import BarbePerturber
import pandas as pd
import pickle


def produce_ranges(data):
    feature_names = data.columns
    feature_range = [np.unique(data[feature]) if len(list(np.unique(data[feature]))) <= 10 or
                                                 np.isscalar(np.unique(data[feature])) else (np.min(data[feature]),
                                                                                             np.max(data[feature]))
                     for feature in feature_names]
    return feature_range


def open_input_file(input_file, file_name):

    # check that the file opens into pandas, extract and return important
    #  rendering info: data, features, types, vals/range
    data = pd.read_csv(input_file, index_col=0)

    if file_name.endswith('.data'):
        feature_names = [str(i) for i in range(len(data.columns))]
        data = data.set_axis(feature_names, axis=1)

    feature_names = list(data)
    # data = data.dropna()
    feature_types = [type(data.iloc[0][feature]) for feature in feature_names]
    #for feature in feature_names:
    #    if len(list(np.unique(data[feature].astype(str)))) <= 10 and not np.all(np.isreal(list(data[feature]))):
    #        data[feature] = data[feature].astype(str)

    temp_perturber = BarbePerturber(training_data=data,
                                    dev_scaling_factor=1,
                                    uniform_training_range=False,
                                    df=None)
    feature_scale = np.round(temp_perturber.get_scale(), 2)
    feature_categories = temp_perturber.get_discrete_values()

    feature_range = [np.unique(data[feature_names[i]]).astype(str) if i in feature_categories.keys() else (np.round(np.nanmin(data[feature_names[i]]), 2),
                                                                                                           np.round(np.nanmax(data[feature_names[i]]), 2))
                     for i in range(len(feature_names))]
    print("IAIN PLEASE: ", feature_categories)
    return (data,
            feature_names,
            feature_types,
            feature_range,
            feature_scale,
            feature_categories)


def open_input_model(input_file, file_name):
    #print(input_file.name)
    input_file = open(input_file, "rb")
    return BlackBoxWrapper(pickle.load(input_file))
    #return None


def check_settings(settings):
    setting_change = None
    if (float(settings['dev_scaling_factor']) % 1 != 0 or
            float(settings['n_perturbations']) % 1 != 0 or
            float(settings['n_bins']) % 1 != 0):
        setting_change = list()
        setting_change.append("Scaling Factor" if settings['dev_scaling_factor'] % 1 != 0 else "")
        setting_change.append("Number Perturbations" if settings['n_perturbations'] % 1 != 0 else "")
        setting_change.append("Number Bins" if settings['n_bins'] % 1 != 0 else "")
        return " ".join(setting_change)
    return setting_change


def fit_barbe_explainer(scales, features, categories, data_row, predictor, indicator_file,
                        settings=None):
    if settings is None:
        settings = {'perturbation_type': 'uniform',
                    'dev_scaling_factor': 5,
                    'input_sets_class': True,
                    'n_perturbations': 5000,
                    'n_bins': 5}
    # IAIN fix error that occurs with odd cases passed as data (seems to error in the call)
    # check if this is data or given ranges instead
    try:
        explainer = BARBE(input_scale=scales,
                          feature_names=features,
                          input_categories=categories,
                          verbose=False,
                          input_sets_class=settings['input_sets_class'],
                          perturbation_type=settings['perturbation_type'],
                          dev_scaling_factor=settings['dev_scaling_factor'],
                          n_perturbations=settings['n_perturbations'],
                          n_bins=settings['n_bins'])
        explanation = explainer.explain(data_row, predictor, ignore_errors=True)
    except ValueError:
        # ValueErrors are the ones we usually handle
        return None, None
    return explainer, explanation


def barbe_rules_table(barbe_rules):
    return pd.DataFrame(barbe_rules, columns=['Text', 'Class', 'Con', 'Supp', 'p_val']).sort_values(by=["p_val"], ascending=True)


def feature_importance_barplot(importance):
    #importance = barbe_instance.get_features(data_row, data_label)
    importance = pd.DataFrame(importance, columns=['Feature', 'Importance'])
    importance['color'] = ['red' if importance.iloc[i]['Importance'] <= 0 else 'green' for i in range(importance.shape[0])]
    my_palette = {'red': 'red', 'green': 'green'}
    plot = sns.barplot(importance, y='Feature', x='Importance', hue='color', palette=my_palette)
    plot.legend_.remove()
    return plot