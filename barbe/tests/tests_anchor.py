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
from barbe.utils.anchor_interface import AnchorExplainer

def test_training_anchor():
    data = pd.read_csv("../dataset/train_loan_raw.csv")
    data = data.drop('Loan_ID', axis=1)
    print(list(data))
    data = data.dropna()

    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    data = data.dropna()
    data = data.sample(frac=1, random_state=117283)

    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Loan_Status']:
            data[cat] = data[cat].astype(float)

    y = data['Loan_Status']
    data = data.drop(['Loan_Status'], axis=1)

    fc = FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=2)

    fc.fit(data, y)

    disc = CategoricalEncoder(ordinal_encoding=True)
    X = disc.fit_transform(training_data=data.copy())

    categorical_names = dict()
    for i in range(len(data.columns)):
        if data.columns[i] in categorical_features:
            categorical_names[i] = disc.get_encoder_key()[data.columns[i]]

    print(categorical_names)

    anchor_exp = AnchorTabularExplainer([0, 1],
                                        list(X.columns),
                                        X.iloc[10:].to_numpy(),
                                        categorical_names)
    exp = anchor_exp.explain_instance(X.to_numpy()[0].reshape((-1, 1)), lambda x: fc.predict(disc.inverse_transform(x)),
                                      threshold=0.95)
    print('Anchor: %s' % (' AND '.join(exp.names())))

    print(confusion_matrix(fc._le.transform(y), fc.predict(data)))

    pass


def test_training_anchor_wrapper():
    data = pd.read_csv("../dataset/train_loan_raw.csv")
    data = data.drop('Loan_ID', axis=1)
    print(list(data))
    data = data.dropna()

    encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    data = data.dropna()
    data = data.sample(frac=1, random_state=117283)

    for cat in categorical_features:
        data[cat] = data[cat].astype(str)

    for cat in list(data):
        if cat not in categorical_features + ['Loan_Status']:
            data[cat] = data[cat].astype(float)

    y = data['Loan_Status']
    data = data.drop(['Loan_Status'], axis=1)

    fc = FIEAPClassifier(protected_feature='Gender=Male', privileged_group=1, unprivileged_group=0, num_clusters=2)

    fc.fit(data, y)

    anchor_exp = AnchorExplainer(data.iloc[10:].copy(), y)
    exp = anchor_exp.explain_instance(data.iloc[0:1], fc, threshold=0.95)
    #exp = anchor_exp.explain_instance(X.to_numpy()[0].reshape((-1, 1)), lambda x: fc.predict(disc.inverse_transform(x)),
    #                                  threshold=0.95)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print(exp.names())

    labels = np.unique(fc.predict(data))
    anchor_preds = anchor_exp.predict(data)
    new_preds = anchor_preds.copy()
    new_preds[anchor_preds] = labels[0]
    new_preds[~anchor_preds] = labels[1]

    #print(new_preds)
    print(confusion_matrix(fc.predict(data), list(new_preds)))
    print(fc.predict(data)[0], list(new_preds)[0])
    print(confusion_matrix(fc._le.transform(y), fc.predict(data)))

    pass