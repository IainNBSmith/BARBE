# IAIN provides plotting utilities to the visualizer
import numpy as np
import seaborn as sns
import matplotlib as mtp
from barbe.explainer import BARBE
from barbe.utils.bbmodel_interface import BlackBoxWrapper
from barbe.perturber import BarbePerturber
import pandas as pd
import pickle

# TODO: fix new error when n_bins=10 and dev_scaling=2 (only in the visualize for some reason)


def get_relevant_label(val):
    if (val < 0.2) and (val > -0.2):
        return "."
    elif (val >= 0.2) and (val < 0.6):
        return "+"
    elif val >= 0.6:
        return "++"
    elif (val <= -0.2) and (val > -0.6):
        return "-"
    elif val <= -0.6:
        return "--"
    return "?"


def get_central_value(str_val):
    str_translation = {'++': 1, '+': 0.5, '.': 0, '-': -0.5, '--': -1}
    return str_translation[str_val]


def category_identifier(ranges):
    return [(len(ranges[i]) == 2 and not isinstance(ranges[i][0], str)) for i in range(len(ranges))]


def correct_scales(org_scales, settings, cat_identifier):
    if settings is None:
        return [org_scales[i] + 1e-10 for i in range(len(cat_identifier))]
    new_scales = []
    for i in range(len(cat_identifier)):
        settings_identifier = 'feature_' + str(i) + "_settings"
        if cat_identifier[i]:
            new_scales.append(settings[settings_identifier]['scale'] + 1e-10)
        else:
            new_scales.append(org_scales[i] + 1e-10)
    return new_scales


def correct_categories(org_categories, settings, cat_identifier, ui):
    if settings is None:
        return org_categories
    new_categories = org_categories.copy()
    for i in range(len(cat_identifier)):
        if not cat_identifier[i]:
            settings_identifier = 'feature_' + str(i) + "_settings"
            # IAIN find the shape of categories

            new_translation = {}
            ui.notification_show(str(i))
            for key in new_categories[i]:
                ui.notification_show(str(key))
                ui.notification_show(str(settings_identifier))
                if key in settings[settings_identifier]['possible_values']:
                    ui.notification_show(str(new_categories))
                    ui.notification_show(str(new_categories[i]))
                    new_translation[key] = new_categories[i][key]

            new_categories[i] = new_translation

    return new_categories


def correct_covariance(org_covariance, corr_scale, settings, cat_identifier):
    if settings is None:
        return org_covariance
    new_cov = org_covariance.copy()
    for x in range(len(cat_identifier)):
        for y in range(len(cat_identifier)):
            new_cov[x, y] = get_central_value(settings['cov'][x][y]) if x != y else corr_scale[x]
    return new_cov


def correct_bounds(settings, cat_identifier):
    if settings is None:
        return [None for _ in range(len(cat_identifier))]
    formatted_bounds = []
    for i in range(len(cat_identifier)):
        settings_identifier = 'feature_' + str(i) + "_settings"
        if cat_identifier[i]:
            formatted_bounds.append(settings[settings_identifier]['bounds'])
        else:
            formatted_bounds.append(None)
    return formatted_bounds


def correct_cov_values(cov):
    cov_replacement = []
    for y in range(cov.shape[1]):
        temp_replacement = []
        for x in range(cov.shape[0]):
            temp_replacement.append(get_relevant_label(cov[x,y]))
        cov_replacement.append(temp_replacement)
    return cov_replacement


def get_next_label(symbol):
    symbol_next_dictionary = {'+': '++', '++': '--', '--': '-', '-': '.', '.': '+'}
    return symbol_next_dictionary[symbol]


def produce_ranges(data, feature_categories):
    feature_names = list(data)
    feature_range = [np.unique(data[feature_names[i]]).astype(str) if i in feature_categories.keys() else (
        np.round(np.nanmin(data[feature_names[i]]), 2),
        np.round(np.nanmax(data[feature_names[i]]), 2))
                     for i in range(len(feature_names))]
    return feature_range


# IAIN TODO: make it so it can detect if a file is data or a ranges file
# TODO: fix error when selecting iris and open twice (maybe some info is left over??)
def open_input_file(input_file, file_name):

    # check that the file opens into pandas, extract and return important
    #  rendering info: data, features, types, vals/range
    data = pd.read_csv(input_file, index_col=0)

    if file_name.endswith('.data'):
        feature_names = [str(i) for i in range(len(data.columns))]
        data = data.set_axis(feature_names, axis=1)

    feature_names = list(data)
    # data = data.dropna()
    # IAIN import line to fix float value issues
    for feature in feature_names:
        if isinstance(data.iloc[0][feature], float):
            if all(data[feature] % 1 == 0):
                data[feature] = data[feature].astype(int)
    #    if len(list(np.unique(data[feature].astype(str)))) <= 10 and not np.all(np.isreal(list(data[feature]))):
    #        data[feature] = data[feature].astype(str)
    feature_types = [type(data.iloc[0][feature]) for feature in feature_names]

    temp_perturber = BarbePerturber(training_data=data,
                                    dev_scaling_factor=1,
                                    uniform_training_range=False,
                                    df=None)
    feature_scale = np.round(temp_perturber.get_scale(), 2)
    feature_cov = np.round(temp_perturber.get_cov(), 2)
    feature_categories = temp_perturber.get_discrete_values()

    feature_range = produce_ranges(data, feature_categories)
    return (data,
            feature_names,
            feature_types,
            feature_range,
            feature_scale,
            feature_cov,
            feature_categories)


def open_input_model(input_file, file_name):
    #print(input_file.name)
    input_file = open(input_file, "rb")
    return BlackBoxWrapper(pickle.load(input_file))
    #return None

'''
    for i in range(len(ranges)):
        try:
            if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
                if (settings['input_bounds'][i] is not None and
                    (settings['input_bounds'][i][0] is not None and
                     data[features[i]] < settings['input_bounds'][i][0]) or
                    (settings['input_bounds'][i][1] is not None and
                     data[features[i]] > settings['input_bounds'][i][1])):
                    setting_change.append("Input outside bounds in " + features[i])
                if settings['new_scale'][i] is not None and settings['new_scale'][i] < 0:
                    setting_change.append("Scale for " + features[i])
            else:
                if data[features[i]] not in settings['modified_category'][i]:
                    setting_change.append("Input outside values in " + features[i])
        except Exception as e:
            return str(ranges) + str(e)
'''


def check_settings(settings, data, ranges, features):
    setting_change = list()
    if (float(settings['dev_scaling_factor']) % 1 != 0 or
            float(settings['n_perturbations']) % 1 != 0 or
            float(settings['n_bins']) % 1 != 0):
        setting_change.append("Scaling Factor" if settings['dev_scaling_factor'] % 1 != 0 else "")
        setting_change.append("Number Perturbations" if settings['n_perturbations'] % 1 != 0 else "")
        setting_change.append("Number Bins" if settings['n_bins'] % 1 != 0 else "")
    if len(setting_change) == 0:
        return None
    return " ".join(setting_change)


def fit_barbe_explainer(scales, features, categories, covariance, bounds, data_row, predictor, indicator_file,
                        settings=None):
    if settings is None:
        settings = {'perturbation_type': 'uniform',
                    'dev_scaling_factor': 2,
                    'input_sets_class': True,
                    'n_perturbations': 5000,
                    'n_bins': 10}
    # IAIN fix error that occurs with odd cases passed as data (seems to error in the call)
    # check if this is data or given ranges instead
    try:
        explainer = BARBE(input_scale=scales,
                          feature_names=features,
                          input_categories=categories,
                          input_bounds=bounds,
                          input_covariance=covariance,
                          verbose=False,
                          input_sets_class=settings['input_sets_class'],
                          perturbation_type=settings['perturbation_type'],
                          dev_scaling_factor=settings['dev_scaling_factor'],
                          n_perturbations=settings['n_perturbations'],
                          n_bins=settings['n_bins'])
        explanation = explainer.explain(data_row, predictor, ignore_errors=True)
    except ValueError as e:
        print(e)
        print(data_row)
        # ValueErrors are the ones we usually handle
        return str(e), None
    return explainer, explanation


def extract_advanced_settings(adv_settings, ranges):
    if adv_settings is None:
        return ([None for _ in range(len(ranges))],
                [None for _ in range(len(ranges))],
                [None for _ in range(len(ranges))],
                [True for _ in range(len(ranges))])
    input_bounds = []
    modified_category = []
    new_scale = []
    change_category = []
    for i in range(len(ranges)):
        settings_identifier = "feature_" + str(i) + "_settings"
        if len(ranges[i]) == 2 and not isinstance(ranges[i][0], str):
            modified_category.append(None)
            change_category.append(None)
            new_scale.append(adv_settings[settings_identifier]['scale'])
            input_bounds.append(adv_settings[settings_identifier]['bounds'])
        else:
            modified_category.append(adv_settings[settings_identifier]['possible_values'])
            change_category.append(adv_settings[settings_identifier]['change'])
            new_scale.append(None)
            input_bounds.append(None)

    return input_bounds, modified_category, new_scale, change_category


def format_bounds(bounds_string):
    if bounds_string is None:
        return None
    format_string = bounds_string.replace(",", " ")
    format_string = format_string.replace("(", "")
    format_string = format_string.replace(")", "").lower()
    if format_string == 'none' or format_string.replace(" ", '') == '':
        return None

    formatted_list = format_string.split(" ")
    if '' in formatted_list:
        formatted_list.remove('')
    if len(formatted_list) != 2:
        return "format for bounds must be one of the following '', None, (None, Number), or Number Number"

    min_b, max_b = formatted_list
    try:
        if min_b != 'none':
            float(min_b)
        if max_b != 'none':
            float(max_b)
        num_check = True
    except ValueError:
        num_check = False
    if not num_check:
        # Throw an error message
        return "numbers must be passed for bounds only (,.) are allowed aside from numerics characters."
    if float(min_b) > float(max_b):
        return "minimum value must be smaller than maximum value in the bound."

    return float(min_b) if min_b != 'none' else None, float(max_b) if max_b != 'none' else None


def barbe_rules_table(barbe_rules):
    return pd.DataFrame(barbe_rules, columns=['Rule', 'Class', 'Support', 'Confidence', 'P-Value']).sort_values(by=["P-Value"], ascending=True)


def barbe_counter_rules_table(counter_rules):
    return pd.DataFrame(counter_rules, columns=['Original Rule', 'New Rule'])


def feature_importance_barplot(importance):
    #importance = barbe_instance.get_features(data_row, data_label)
    importance = pd.DataFrame(importance, columns=['Feature', 'Importance'])
    importance['color'] = ['red' if importance.iloc[i]['Importance'] <= 0 else 'green' for i in range(importance.shape[0])]
    my_palette = {'red': 'red', 'green': 'green'}
    plot = sns.barplot(importance, y='Feature', x='Importance', hue='color', palette=my_palette)
    plot.legend_.remove()
    return plot