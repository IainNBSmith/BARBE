from datetime import datetime

import numpy as np

from barbe.tests.tests import _get_data
from barbe.perturber import BarbePerturber


def test_produce_barbe_perturbations(n_perturbations=5000):
    # From this test we learned that a sample must be discretized into bins
    #  then it has scale assigned by the training sample and only then can it
    #  be perturbed

    training_data, _ = _get_data()
    data_row = training_data.drop('class', axis=1).iloc[0]

    print("Running test: BARBE Perturbation")
    start_time = datetime.now()
    bp = BarbePerturber(training_data.drop('class', axis=1))
    perturbed_data = bp.produce_perturbation(n_perturbations, data_row=data_row)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(perturbed_data)

    print(bp.get_discrete_values())


def test_categorical_perturbation(n_perturbations=5000):
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

    print("Running test: BARBE Perturbations Categorical")
    start_time = datetime.now()
    bp = BarbePerturber(training_data.drop('class', axis=1))
    perturbed_data = bp.produce_perturbation(n_perturbations, data_row=data_row)
    print("Test Time: ", datetime.now() - start_time)
    print(data_row)
    print(perturbed_data)

    print(bp.get_discrete_values())
