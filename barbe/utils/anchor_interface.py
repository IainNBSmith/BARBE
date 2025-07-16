# TODO: get code and add anchor w/ interface
from anchor.anchor_tabular import AnchorTabularExplainer
from barbe.discretizer import CategoricalEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re


class AnchorExplainer:
    def __init__(self, data, y):
        self.encoder = CategoricalEncoder(ordinal_encoding=True)
        self.labels = LabelEncoder()
        class_names = np.unique(self.labels.fit_transform(y))

        data = self.encoder.fit_transform(training_data=data)

        categorical_names = dict()
        named_columns = list(data.columns)
        current_keys = self.encoder.get_encoder_key()
        for key in current_keys.keys():
            if key not in self.encoder._finite_numeric_features:
                categorical_names[named_columns.index(key)] = current_keys[key]

        print(categorical_names)

        self.explainer = AnchorTabularExplainer(list(class_names),
                                                named_columns,
                                                data.to_numpy(),
                                                categorical_names)
        self.current_explanation = None

    def explain_instance(self, instance, bbmodel, **kwargs):
        predict_function = lambda x: self.labels.inverse_transform(bbmodel.predict(self.encoder.inverse_transform(x)))
        self.current_explanation = self.explainer.explain_instance(self.encoder.transform(instance).to_numpy().reshape((-1, 1)),
                                                                   predict_function,
                                                                   **kwargs)
        return self.current_explanation

    def predict(self, X):
        current_features = self.current_explanation.names()
        all_evaluations = True
        for rule in current_features:
            eval_string = ""
            parts = rule.split(" ")
            if "=" in parts:
                eval_string = f"X['{parts[0]}'] == '{parts[2]}'"
            elif "<" in parts and "<=" in parts:
                eval_string = f"({parts[0]} <= X['{parts[2]}']) & (X['{parts[2]}'] < {parts[4]})"
            elif "<=" in parts:
                eval_string = f"(X['{parts[0]}'] <= {parts[2]})"
            elif ">" in parts:
                eval_string = f"(X['{parts[0]}'] > {parts[2]})"
            else:
                print("OOPS")

            print(eval_string)
            all_evaluations = all_evaluations & eval(eval_string)

        return all_evaluations


