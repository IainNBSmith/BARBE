from datetime import datetime
from barbe.tests.tests import _get_data
from barbe.utils.sigdirect_interface import SigDirectWrapper
import numpy as np


def test_categorical_sigdirect(n_perturbations=5000):
    # Currently no implementation of categorical variables into the perturber, this should be fixed
    # July 17th TODO: implement categorical variables into pertruber, these will be detected then set to numeric values
    #  TODO: finally assigned by checking the numerical value at the end and translating
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed
    # TODO: for discrete labels should we make it even odds or somehow change how perturbing is done?
    # TODO: ensure current sigdirect wrapper works with categorical values

    def categorical_named(x):
        if x <= 1:
            return "A"
        elif x <= 3:
            return "B"
        return "C"

    training_data, _ = _get_data()
    # make a discrete value
    # boolean
    training_data[list(training_data)[0]] = training_data[list(training_data)[0]] < 1
    # string
    training_data[list(training_data)[1]] = training_data[list(training_data)[1]].apply(categorical_named)
    # float / int
    training_data[list(training_data)[2]] = np.ceil(training_data[list(training_data)[2]])
    data_row = training_data.drop('class', axis=1).iloc[0]
    pred_feature = training_data['class']
    training_data = training_data.drop('class', axis=1)

    print("Running test: SigDirect Categorical")
    start_time = datetime.now()
    clas = SigDirectWrapper(list(training_data))
    clas.fit(training_data.to_numpy(), pred_feature)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(clas.get_features(data_row, 2))
    print(clas.get_all_rules())