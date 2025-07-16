import numpy as np

from barbe.utils.fieap_interface import FIEAPClassifier
from barbe.tests.tests import _get_data
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from anchor.anchor_tabular import AnchorTabularExplainer
from barbe.discretizer import CategoricalEncoder
from barbe.utils.bbmodel_interface import BlackBoxWrapper
from barbe.utils.anchor_interface import AnchorExplainer
import dice_ml
import dill


def test_training_dice():
    data = pd.read_csv("../dataset/ibm_hr_attrition.csv")
    #data = data.drop('Loan_ID', axis=1)
    data = data.drop(['StandardHours', 'EmployeeCount', 'Over18'], axis=1)
    print(list(data))
    data = data.dropna()

    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                            'OverTime']

    data = data.dropna()
    data = data.sample(frac=1, random_state=5771)

    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Attrition']:
            data[cat] = data[cat].astype(float)

    y = data['Attrition']
    data = data.drop(['Attrition'], axis=1)

    #fc = FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=2)

    with open('../pretrained/attrition_rf.pkl', 'rb') as f:
        fc = dill.load(f)
    #fc.fit(data, y)
    fc = BlackBoxWrapper(bbmodel=fc, class_labels=['Yes', 'No'])

    disc = CategoricalEncoder(ordinal_encoding=True)
    X = disc.fit_transform(training_data=data.copy())

    categorical_names = dict()
    not_categorical = list()
    for i in range(len(data.columns)):
        if data.columns[i] in categorical_features:
            categorical_names[i] = disc.get_encoder_key()[data.columns[i]]
        else:
            not_categorical.append(data.columns[i])

    print(categorical_names)

    data['target'] = y
    print("Getting Data...")
    d = dice_ml.Data(dataframe=data, continuous_features=not_categorical, outcome_name='target')
    print("Getting Model...")
    m = dice_ml.Model(model=fc, backend='sklearn')
    print("Getting Dice...")
    dice_exp = dice_ml.Dice(d, m, method='random')
    print("Getting Counterfactuals...")
    counterfactuals = dice_exp.generate_counterfactuals(data.drop('target', axis=1).iloc[48:49], total_CFs=5, desired_class='opposite')

    print("Feature Importance")
    imp = dice_exp.local_feature_importance(data.drop('target', axis=1).iloc[48:49], total_CFs=10, desired_class='opposite')
    print(imp.local_importance)
    print(fc.predict(data.drop('target', axis=1).iloc[48:49]))
    print("Counterfactual")
    print(counterfactuals.cf_examples_list[0].final_cfs_df.to_numpy())
    counterfactuals.visualize_as_dataframe()
    print(counterfactuals)

    pass