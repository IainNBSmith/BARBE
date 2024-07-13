"""
The point of this code is to handle sigdirect output in a way that would be improper to handle on
 the side of BARBE.

TODO: REMOVE OHE OR IMPLEMENT IT FROM SOMEWHERE ELSE
"""
import itertools
import os
from collections import defaultdict

# from distutils.core import setup
# from Cython.Build import cythonize

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# IAIN requires main folder to contain rule.py
from sigdirect import SigDirect
import numpy as np


class SigDirectWrapper:
    def __init__(self, ohe):
        # TODO IAIN for now keep dummy ohe to remove later

        # IAIN this should have settings for sigdirect and more accurate utilities
        # IAIN intention is for the user to get the same flexibility as scikit
        self._sigdirect_model = SigDirect()
        self._oh_enc = None
        self._l_enc = None
        self._rules = None

    # TODO
    # IAIN the following three functions will take input data and create and encoder
    #  these will also decode data when needed in other functions. The encoder should only
    #  be handled here.
    def _create_encoder(self, X):
        # self._l_enc = LabelEncoder()
        # l_enc_X = self._l_enc.fit_transform(X)
        self._oh_enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
        self._oh_enc.fit(X)

    def _encode(self, X):
        print(self._oh_enc.transform(X).todense())
        return np.asarray(self._oh_enc.transform(X).todense()).astype(int)

    def _decode(self, enc_X):
        pass

    def fit(self, X, y):
        # IAIN sigdirect expects its y as a column number thus
        # IAIN exactly as you would expect from sklearn
        # IAIN this should do data handling that SigDirect does not do

        # TODO
        # IAIN create an encoder and encode the X data the code can be found inside of
        #  barbe.py where __data_inverse_barbe is (it is a nearly unnecessary step)
        self._create_encoder(X)
        self._sigdirect_model.fit(self._encode(X), y.tolist())

    def predict(self, X):
        # IAIN holy shit why does this work?? Hence the requirement of a wrapper...

        # TODO
        # IAIN use the encoder to make input X into the encoded data
        self._sigdirect_model.predict(self._encode(X))
        # self._sigdirect_model.predict(X.reshape((1, -1)), 2).astype(int)

    def _generate_rules(self, data_row, true_label):
        all_rules = defaultdict(list)
        all_raw_rules = self._sigdirect_model.get_all_rules()
        # TODO: why is this only one rule??
        print("IAIN all raw rules")
        print(all_raw_rules)
        predicted_label = self._sigdirect_model.predict(data_row.reshape((1, -1)), 2).astype(int)

        # convert raw rules to rules (one-hot-decoding them)
        if predicted_label[0] == true_label:
            print("makin bacon")
            for x, y in all_raw_rules.items():
                all_rules[x] = [(t, self._ohe, data_row) for t in y]

        else:
            predicted_label = -1  # to show we couldn't predict it correctly

        self._rules = all_rules
        return predicted_label

    def get_features(self, data_row, true_label):
        # IAIN as originally written
        """
        Input: all_rules ()  ->
               true_label () ->
        Purpose: use applied rules first, and then the rest of the applicable rules, and then all rules (other labels,
         rest of them match)
        Output: feature_value_pairs () ->
        """
        label_quality = self._generate_rules(data_row, true_label)
        print(label_quality)
        all_rules = self._rules
        print(all_rules)
        print(true_label)

        # return (0, feature_value_pairs, prediction_score, predicted_label)

        # applied rules,
        applied_sorted_rules = sorted(all_rules[true_label],
                                      key=lambda x: (
                                          len(x[0].get_items()),
                                          - x[0].get_confidence() * x[0].get_support(),
                                          x[0].get_log_p(),
                                          - x[0].get_support(),
                                          -x[0].get_confidence(),
                                      ),
                                      reverse=False)

        # applicable rules, except the ones in applied rules.
        applicable_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x != true_label]),
                                         key=lambda x: (
                                             len(x[0].get_items()),
                                             - x[0].get_confidence() * x[0].get_support(),
                                             x[0].get_log_p(),
                                             - x[0].get_support(),
                                             -x[0].get_confidence(),
                                         ),
                                         reverse=False)

        # all rules, except the ones in applied rules.
        other_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x != true_label]),
                                    key=lambda x: (
                                        len(x[0].get_items()),
                                        - x[0].get_confidence() * x[0].get_support(),
                                        x[0].get_log_p(),
                                        - x[0].get_support(),
                                        -x[0].get_confidence(),
                                    ),
                                    reverse=False)

        counter = len(all_rules)
        bb_features = defaultdict(int)

        # First add applied rules
        applied_rules = []
        for rule, ohe, original_point_sd in applied_sorted_rules:
            temp = np.zeros(original_point_sd.shape[0]).astype(int)
            temp[rule.get_items()] = 1
            if np.sum(temp & original_point_sd.astype(int)) != temp.sum():
                continue
            else:
                applied_rules.append(rule)
            rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
            #         rule_items = temp ## TEXT (uncomment for TEXT)
            for item, val in enumerate(rule_items):
                if val is None:
                    continue
                #                 if val==0: ## TEXT (uncomment for TEXT)
                #                     continue ## TEXT (uncomment for TEXT)
                #                 if item not in bb_features:
                bb_features[item] += rule.get_support()
            #                     bb_features[item] += counter
            #                 bb_features[item] = max(bb_features[item],  rule.get_confidence()/len(rule.get_items()))
            counter -= 1
        set_size_1 = len(bb_features)

        # Second, add applicable rules
        applicable_rules = []
        for rule, ohe, original_point_sd in applicable_sorted_rules:
            temp = np.zeros(original_point_sd.shape[0]).astype(int)
            temp[rule.get_items()] = 1
            if np.sum(temp & original_point_sd.astype(int)) != temp.sum():
                continue
            else:
                applicable_rules.append(rule)
            rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
            #         rule_items = temp ## TEXT (uncomment for TEXT)
            for item, val in enumerate(rule_items):
                if val is None:
                    continue
                if item not in bb_features:
                    #                 bb_features[item] += rule.get_support()
                    bb_features[item] += counter
            counter -= 1

        # Third, add other rules.
        other_rules = []
        for rule, ohe, original_point_sd in other_sorted_rules:
            temp = np.zeros(original_point_sd.shape[0]).astype(int)
            temp[rule.get_items()] = 1
            # avoid applicable rules
            if np.array_equal(temp, temp & original_point_sd.astype(int)):  # error??? it was orig...[0].astype
                continue
            #             elif temp.sum()==1:
            #                 continue
            elif temp.sum() - np.sum(temp & original_point_sd.astype(int)) > 1:  # error???
                continue
            #             else:
            rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
            #         rule_items = temp ## TEXT (uncomment for TEXT)
            seen_set = 0
            for item, val in enumerate(rule_items):
                if val is None:
                    continue
                if item not in bb_features:
                    #                 bb_features[item] += rule.get_support()
                    #                     bb_features[item] += counter
                    candid_feature = item
                    pass
                else:
                    seen_set += 1
            if seen_set == temp.sum() - 1:  # and (item not in bb_features):
                bb_features[candid_feature] += counter
                other_rules.append(rule)
            counter -= 1

        print(bb_features)
        feature_value_pairs = sorted(bb_features.items(), key=lambda x: x[1], reverse=True)

        return feature_value_pairs, None

    def get_translation(self):
        pass
