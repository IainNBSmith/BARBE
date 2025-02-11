from barbe.utils.evaluation_measures import *
from barbe.utils.bbmodel_interface import *
from barbe.discretizer import CategoricalEncoder
from sklearn.metrics import accuracy_score
from barbe.perturber import ClassBalancedPerturber


class DummyExplainer:

    def __init__(self, training_data):
        self._predict_class = None
        self._features = list(training_data)
        self.encoder = CategoricalEncoder()
        self.encoder.fit(training_data=training_data)
        self.perturber = ClassBalancedPerturber(training_data=training_data)

    def predict(self, X):
        return np.repeat(self._predict_class, X.shape[0])

    def explain(self, input_data, bbmodel):
        input_row = pd.DataFrame(columns=self._features, index=[0])
        # print(pert_row)
        # print(list(iris_training))
        input_row.iloc[0] = input_data.to_numpy().reshape((1, -1))
        self._predict_class = bbmodel.predict(input_row)[0]
        self.perturber.produce_balanced_perturbation(1000,
                                                     BlackBoxWrapper(bbmodel),
                                                     data_row=input_data)
        return None

    def get_surrogate_fidelity(self, comparison_model=None, comparison_data=None,
                               comparison_method=accuracy_score, weights=None, original_data=None):
        wrapped_comparison = BlackBoxWrapper(comparison_model)
        if self.encoder is not None:
            discretize_call = self.encoder.transform
        else:
            discretize_call = lambda x: x
        if weights is not None and weights in 'euclidean':
            if comparison_data is None:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(self.perturbed_data))
            else:
                weights = euclidean_weights(discretize_call(original_data),
                                            discretize_call(comparison_data).to_numpy())
        elif weights is not None and weights in 'nearest-neighbors':
            if comparison_data is None:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(self.perturbed_data))
            else:
                weights = nearest_neighbor_weights(discretize_call(original_data),
                                                   discretize_call(comparison_data).to_numpy())
        # IAIN check if comparison model, data, and method is f(a,b) is comparing vectors
        # IAIN compare the surrogate to the original input model
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN set default and some alternative options for comparison of classifications
        # IAIN comparison_method(y_true, y_pred)
        if (comparison_model is None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        #elif (comparison_model is None) and (comparison_data is not None):
        #    return comparison_method(self._blackbox_classification['perturbed'],
        #                             self._surrogate_classification['perturbed'],
        #                             sample_weight=weights)

        elif (comparison_model is not None) and (comparison_data is None):
            return comparison_method(wrapped_comparison.predict(self.perturbed_data),
                                     self.predict(self.perturbed_data),
                                     sample_weight=weights)
        elif (comparison_model is not None) and (comparison_data is not None):
            return comparison_method(wrapped_comparison.predict(comparison_data),
                                     self.predict(comparison_data),
                                     sample_weight=weights)