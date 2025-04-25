import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from barbe.explainer import BARBE

import seaborn as sns
import matplotlib.pyplot as plt

def distribution_visualization_loan():
    # example of where lime1 fails
    # lime1 can only explain pre-processed data (pipeline must be separate and interpretable from model)
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

    preprocess = ColumnTransformer([('enc', encoder, categorical_features)], remainder='passthrough')
    model = Pipeline([('pre', preprocess),
                      ('clf', MLPClassifier(hidden_layer_sizes=(100, 50,), random_state=301257))])

    split_point_test = int((data.shape[0] * 0.2) // 1)  # 80-20 split
    loan_perturb = data.iloc[0:50].reset_index()
    loan_test = data.iloc[0:split_point_test].reset_index()
    loan_training = data.iloc[split_point_test:].reset_index()
    loan_training_label = y[split_point_test:]

    loan_training_stop = int(loan_training.shape[0] // 3)

    for feature in list(loan_training):
        unique_values = np.unique(loan_training.iloc[0:loan_training_stop][feature])
        print(feature, type(unique_values[0]))
        if isinstance(unique_values[0], str):
            loan_perturb[feature] = [(value if value in unique_values else "unknown") for value in loan_perturb[feature]]
            loan_test[feature] = [(value if value in unique_values else "unknown") for value in loan_test[feature]]
            loan_training[feature] = [(value if value in unique_values else "unknown") for value in loan_training[feature]]

    loan_perturb = loan_perturb.drop(['index'], axis=1)
    loan_test = loan_test.drop(['index'], axis=1)
    loan_training = loan_training.drop(['index'], axis=1)

    model.fit(loan_training, loan_training_label)

    sns.displot(loan_test, x='LoanAmount', y='ApplicantIncome', kind='kde', hue=model.predict(loan_test),
                hue_order=['N', 'Y'], fill=True, palette={'Y':(0,158/255,115/255, 1), 'N':(213/255,94/255,0,1)}, alpha=0.75, levels=5, hatch={'Y': '///', 'N': '*'})#, hue=loan_training_label)
    plt.scatter(loan_perturb.iloc[2]['LoanAmount'], loan_perturb.iloc[2]['ApplicantIncome'], color=(230/255,159/255,0,1))
    plt.xlim((-100, 600))
    plt.ylim((-10000, 21000))
    plt.show()

    train_exp = loan_training.sample(frac=1/2)
    explainer = BARBE(training_data=train_exp,
                      input_bounds=None,  # [(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                      perturbation_type='normal',
                      n_perturbations=2000,
                      dev_scaling_factor=4,
                      n_bins=5,
                      verbose=False,
                      input_sets_class=False, balance_classes=False)
    explainer.explain(loan_perturb.iloc[2], model)

    loan_perts = explainer.get_perturbed_data()

    explainer = BARBE(training_data=train_exp,
                      input_bounds=None,  # [(4.4, 7.7), (2.2, 4.4), (1.2, 6.9), (0.1, 2.5)],
                      perturbation_type='standard-normal',
                      n_perturbations=2000,
                      dev_scaling_factor=2,
                      higher_frequent_category_odds=False,
                      n_bins=5,
                      verbose=False,
                      input_sets_class=False, balance_classes=False)
    explainer.DEFAULT_CATEGORY_FREQUENCY_ODDS = False

    explainer.explain(loan_perturb.iloc[2], model)
    loan_perts_2 = explainer.get_perturbed_data()

    loan_all = pd.concat([loan_training, loan_perts, loan_perts_2], ignore_index=True)
    loan_all['sources'] = 'Original'
    cnt_og = loan_training.shape[0]
    cnt_mult = loan_perts.shape[0]
    cnt_norm = loan_perts_2.shape[0]
    loan_all.loc[loan_all.index[cnt_og:(cnt_og+cnt_mult)], 'sources'] = 'Multivariate'
    loan_all.loc[loan_all.index[(cnt_og+cnt_mult):], 'sources'] = 'Normal'


    #sns.displot(loan_all, x='LoanAmount', y='ApplicantIncome', kind='kde', hue='sources',
    #            hue_order=['N', 'Y'], fill=True, palette={'Y':(0,158/255,115/255, 1), 'N':(213/255,94/255,0,1)}, alpha=0.75, levels=3, hatch={'Y': '///', 'N': '*'})#, hue=loan_training_label)
    #plt.scatter(loan_perturb.iloc[2]['LoanAmount'], loan_perturb.iloc[2]['ApplicantIncome'], color=(230/255,159/255,0,1))
    #plt.xlim((-100, 600))
    #plt.ylim((-10000, 21000))
    #plt.show()

    sns.displot(loan_perts, x='LoanAmount', y='ApplicantIncome', kind='kde', hue=model.predict(loan_perts),
                hue_order=['N', 'Y'], fill=True, palette={'Y':(0,158/255,115/255, 1), 'N':(213/255,94/255,0,1)}, alpha=0.75, levels=5, hatch={'Y': '///', 'N': '*'})#, hue=loan_training_label)
    plt.scatter(loan_perturb.iloc[2]['LoanAmount'], loan_perturb.iloc[2]['ApplicantIncome'], color=(230/255,159/255,0,1))  # , hue=loan_training_label)
    plt.xlim((-100, 600))
    plt.ylim((-10000, 21000))
    plt.show()

    sns.displot(loan_perts_2, x='LoanAmount', y='ApplicantIncome', kind='kde', hue=model.predict(loan_perts_2),
                hue_order=['N', 'Y'], fill=True, palette={'Y':(0,158/255,115/255, 1), 'N':(213/255,94/255,0,1)}, alpha=0.75, levels=5, hatch={'Y': '///', 'N': '*'})#, hue=loan_training_label)
    plt.scatter(loan_perturb.iloc[2]['LoanAmount'], loan_perturb.iloc[2]['ApplicantIncome'], color=(230/255,159/255,0,1))  # , hue=loan_training_label)
    plt.xlim((-100, 600))
    plt.ylim((-10000, 21000))
    plt.show()


if __name__ == "__main__":
    distribution_visualization_loan()
