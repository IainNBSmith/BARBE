
from barbe.utils.fieap_interface import FIEAPClassifier
from barbe.tests.tests import _get_data
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


def test_training_fieap():
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

    print(confusion_matrix(fc._le.transform(y), fc.predict(data)))

    pass
