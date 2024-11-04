from shiny import run_app
from barbe.utils.visualizer_utils import *

# TODO: Either add NAN handling or add an error message if data with nans is provided
# TODO: what should we do handling max and mins in the context of BARBE
# TODO: try to load ssproj data (remove na's baka)


def test_visualizer():
    run_app("visualizer", app_dir="../")


def test_open_data():
    data, names, types, ranges, _, _ = open_input_file('../dataset/loan_data.csv', '../dataset/loan_data.csv')
    print(data)
    print(names)
    print(types)
    print(ranges)
    print(list(ranges[-1].astype(str)))


def test_open_private():
    data, names, types, ranges, _, _ = open_input_file('../dataset/FOS_data_saved.csv', '../dataset/FOS_data_saved.csv')
    print(data)
    print(names)
    print(types)
    print(ranges)
    print(list(ranges[-1]))


def test_bad_data():
    data, names, types, ranges = open_input_file('../dataset/nope')
    print(data)
    print(names)
    print(types)
    print(ranges)



def test_loan_util():
    # TODO: see why crashes happen in webpage but not here
    #  TODO: based on results here I think it is a formatting innaccury, should pass categorical info from the perturber to be certain
    data = pd.read_csv("../dataset/loan_test.csv", index_col=0)
    print(data.dtypes)
    #y = data['Loan_Status']
    #data = data.drop(['Loan_Status'], axis=1)
    with open("../pretrained/loan_test_decision_tree.pickle", "rb") as f:
        model = pickle.load(f)

    explainer1 = BARBE(training_data=data, input_categories=None, verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')
    explainer = BARBE(training_data=None, input_categories=explainer1.get_perturber('categories'),
                      input_scale=explainer1.get_perturber('scale'),
                      feature_names=list(data),
                      verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')

    data, features, types, ranges, scales, categories = open_input_file("../dataset/loan_test.csv",
                                                                        "../dataset/loan_test.csv")

    print("IAIN COMPARE")
    print(explainer1.get_perturber('categories'))
    print(categories)
    print()
    print(explainer1.get_perturber('scale'))
    print(scales)
    print()
    print(list(data))
    print(features)
    #assert False

    explainer, explanation = fit_barbe_explainer(scales,
                                                  list(data),
                                                  categories,
                                                  data.iloc[0], model, False, settings=None)
    explanation = explainer.explain(data.iloc[0], model)
    # print(confusion_matrix(model.predict(data), y))
    print(model.predict(data))
    print(explainer._surrogate_model.predict(data.to_numpy()))
    print(explainer.get_surrogate_fidelity())
    print(np.unique(explainer._perturbed_data[:, 0]))
    print(explainer.get_available_classes())

    reformat_input = pd.DataFrame(columns=list(data), index=[0])
    reformat_input.iloc[0] = data.iloc[0]

    print(explainer.get_counterfactual_explanation(reformat_input, 'Y'))


def test_iris_util():
    # TODO: see why crashes happen in webpage but not here
    #  TODO: based on results gethere I think it is a formatting innaccury, should pass categorical info from the perturber to be certain
    data = pd.read_csv("../dataset/iris_test.csv", index_col=0)
    print(data.dtypes)
    #y = data['Loan_Status']
    #data = data.drop(['Loan_Status'], axis=1)
    with open("../pretrained/iris_test_decision_tree.pickle", "rb") as f:
        model = pickle.load(f)

    explainer1 = BARBE(training_data=data, input_categories=None, verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')
    explainer = BARBE(training_data=None, input_categories=explainer1.get_perturber('categories'),
                      input_scale=explainer1.get_perturber('scale'),
                      feature_names=list(data),
                      verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')

    data, features, types, ranges, scales, categories = open_input_file("../dataset/loan_test.csv",
                                                                        "../dataset/loan_test.csv")

    print("IAIN COMPARE")
    print(explainer1.get_perturber('categories'))
    print(categories)
    print()
    print(explainer1.get_perturber('scale'))
    print(scales)
    print()
    print(list(data))
    print(features)
    #assert False

    explainer, explanation = fit_barbe_explainer(scales,
                                                  list(data),
                                                  categories,
                                                  data.iloc[0], model, False, settings=None)
    explanation = explainer.explain(data.iloc[0], model)
    # print(confusion_matrix(model.predict(data), y))
    print(model.predict(data))
    print(explainer.get_surrogate_fidelity())
    print(np.unique(explainer._perturbed_data[:, 0]))
    reformat_input = pd.DataFrame(columns=list(data), index=[0])
    reformat_input.iloc[0] = data.iloc[0]

    print(explainer.get_counterfactual_explanation(data.iloc[0], '~Y'))


def test_private_util():
    # TODO: see why crashes happen in webpage but not here
    #  TODO: based on results here I think it is a formatting innaccury, should pass categorical info from the perturber to be certain
    data = pd.read_csv("../dataset/FoS_test_ada_boost.csv", index_col=0)
    print(data.dtypes)
    #y = data['Loan_Status']
    #data = data.drop(['Loan_Status'], axis=1)
    with open("../pretrained/ada_boost_pipeline_model.pkl", "rb") as f:
        model = pickle.load(f)

    explainer1 = BARBE(training_data=data, input_categories=None, verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')
    explainer = BARBE(training_data=None, input_categories=explainer1.get_perturber('categories'),
                      input_scale=explainer1.get_perturber('scale'),
                      feature_names=list(data),
                      verbose=True, input_sets_class=True,
                      dev_scaling_factor=1, perturbation_type='uniform')

    data, features, types, ranges, scales, categories = open_input_file("../dataset/FoS_test_ada_boost.csv",
                                                                        "../dataset/FoS_test_ada_boost.csv")

    print("IAIN COMPARE")
    print(explainer1.get_perturber('categories'))
    print(categories)
    print()
    print(explainer1.get_perturber('scale'))
    print(scales)
    print()
    print(list(data))
    print(features)
    #assert False
    # IAIN broke on using a generated counterfactual (may need to check and output any model errors)
    #  IAIN U is also not in gender for some reason..?

    # TODO: check why some numbers like AP.Age are left the same??
    explainer, explanation = fit_barbe_explainer(scales,
                                                  list(data),
                                                  categories,
                                                  data.iloc[0], model, False, settings=None)
    explanation = explainer.explain(data.iloc[0], model)
    # print(confusion_matrix(model.predict(data), y))
    print(model.predict(data))
    print(explainer._surrogate_model.predict(data.to_numpy()))
    print(explainer.get_surrogate_fidelity())
    print(np.unique(explainer._perturbed_data[:, 0]))
    print(explainer.get_available_classes())

    reformat_input = pd.DataFrame(columns=list(data), index=[0])
    reformat_input.iloc[0] = data.iloc[0]

    print(explainer.get_counterfactual_explanation(reformat_input, 'Y'))