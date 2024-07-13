"""
The point of this code is to handle sigdirect output in a way that would be improper to handle on
 the side of BARBE.

"""
import itertools
import os
from collections import defaultdict

# from distutils.core import setup
# from Cython.Build import cythonize

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer

# IAIN requires main folder to contain rule.py
from sigdirect import SigDirect
import numpy as np


class SigDirectWrapper:
    def __init__(self, column_names, verbose=False):
        # IAIN this should have settings for sigdirect and more accurate utilities
        # IAIN intention is for the user to get the same flexibility as scikit
        # IAIN adjusting settings worked!!
        self._sigdirect_model = SigDirect(
                clf_version=1,
                alpha=0.1,
                early_stopping=False,
                confidence_threshold=0.2,
                is_binary=True,
                get_logs=False,
                other_info=None)
        self._oh_enc = None
        self._rules = None
        self._feature_names = column_names
        self._verbose = verbose
        self._verbose_header = "SigDirect:"

    def _create_encoder(self, X):
        if self._verbose:
            print(self._verbose_header, "on encoder creation:", X.shape, X)
        # is the output importance here??
        # we also along with the bins want the importance of a feature so maybe that is what
        #  they were doing before only just implicitly (check with Osmar)
        self._oh_enc = OneHotEncoder(categories='auto', handle_unknown='ignore',
                                     min_frequency=None)
        self._kb_discrete = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self._kb_discrete.fit(X)
        self._oh_enc.fit(self._kb_discrete.transform(X))

    def _encode(self, X):
        if self._verbose:
            print(self._verbose_header, "before encoding:", X.shape, X)
        return np.asarray(self._oh_enc.transform(self._kb_discrete.transform(X)).todense()).astype(int)

    def _decode(self, enc_X):
        # IAIN generalize this
        partial_decoding = self._oh_enc.inverse_transform(enc_X)[0]
        partial_usable = [i for i in range(len(partial_decoding)) if partial_decoding[i] is not None]
        if self._verbose:
            print(self._verbose_header, "before decoding:", enc_X)
            print(self._verbose_header, "OneHot decode:", partial_decoding)
            print(partial_usable, partial_decoding[partial_usable])
            print(self._kb_discrete.bin_edges_[partial_usable][0])
            print(self._verbose_header, "corresponding bins", self._kb_discrete.bin_edges_[partial_usable][0][partial_decoding[partial_usable].astype(int)])
        # do not need to fully decode I only need to handle which bin value is assigned
        # return self._kb_discrete.inverse_transform(self._oh_enc.inverse_transform(enc_X))
        return_list = [None for i in range(len(partial_decoding))]
        for i in partial_usable:
            print(i, int(partial_decoding[i]))
            print(len(return_list))
            print(len(self._kb_discrete.bin_edges_))
            print(len(self._kb_discrete.bin_edges_[i]))
            return_list[i] = self._kb_discrete.bin_edges_[i][int(partial_decoding[i])]
        # return_list[partial_usable] = self._kb_discrete.bin_edges_[partial_usable][0][partial_decoding[partial_usable].astype(int)][0]
        return [return_list]

    def fit(self, X, y):
        self._create_encoder(X)
        if self._verbose:
            print(self._verbose_header, self._encode(X))
            print(self._verbose_header, "training y:", y.tolist())
        # IAIN something is wrong when fitting sigdirect
        self._sigdirect_model.fit(self._encode(X), y.tolist())

    def predict(self, X):
        return self._sigdirect_model.predict(self._encode(X))

    def _generate_rules(self, data_row, true_label):
        all_rules = defaultdict(list)
        all_raw_rules = self._sigdirect_model.get_all_rules()
        if self._verbose:
            print(self._verbose_header, "all raw rules:", all_raw_rules)
        predicted_label = self.predict(data_row.to_numpy().reshape(1, -1))

        if predicted_label[0] == true_label:
            if self._verbose:
                print(self._verbose_header, "rules matched", predicted_label, true_label)
            for x, y in all_raw_rules.items():
                all_rules[x] = [(t, self._oh_enc, self._encode(data_row.to_numpy().reshape(1,-1))) for t in y]

        else:
            if self._verbose:
                print(self._verbose_header, "rules did not match", predicted_label, true_label)
            predicted_label = -1  # to show we couldn't predict it correctly
        self._rules = all_rules
        return predicted_label

    # def get_all_features(self):
    #     for key, item in self._rules:
    #         print(key, item[0].get_confidence(), item[0].get_support())

    def get_categories(self):
        return self._oh_enc.categories_

    def get_features(self, data_row, true_label):
        # IAIN now one thing to try is when decoding we just check where relevant bins to the sample appear,
        #  I'm pretty sure we would still do decoding for this though
        print("Bins:", self._kb_discrete.bin_edges_)
        label_quality = self._generate_rules(data_row, true_label)
        all_rules = self._rules
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
            temp = np.zeros(original_point_sd.shape[1]).astype(int)
            temp[rule.get_items()] = 1
            if self._verbose:
                print(self._verbose_header, "item of note in applied", rule.get_items())
                print(self._verbose_header, "encoding meaning", self._decode(temp.reshape((1, -1))))

            if np.sum(temp & original_point_sd.astype(int)) != temp.sum():
                continue
            else:
                applied_rules.append(rule)
            # rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
            rule_items = self._decode(temp.reshape((1, -1)))[0]
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
            temp = np.zeros(original_point_sd.shape[1]).astype(int)
            temp[rule.get_items()] = 1
            if self._verbose:
                print(self._verbose_header, "item of note in applicable rules", rule.get_items())
                print(self._verbose_header, "encoding meaning", self._decode(temp.reshape((1, -1))))

            if np.sum(temp & original_point_sd.astype(int)) != temp.sum():
                continue
            else:
                applicable_rules.append(rule)
            rule_items = self._decode(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
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
            temp = np.zeros(original_point_sd.shape[1]).astype(int)
            # tell you where the ohe is encoded
            temp[rule.get_items()] = 1
            if self._verbose:
                print(self._verbose_header, "item of note in other", rule.get_items())
                print(self._verbose_header, "encoding meaning", self._decode(temp.reshape((1, -1))))
            # avoid applicable rules
            if np.array_equal(temp, temp & original_point_sd.astype(int)):  # error??? it was orig...[0].astype
                continue
            #             elif temp.sum()==1:
            #                 continue
            elif temp.sum() - np.sum(temp & original_point_sd.astype(int)) > 1:  # error???
                continue
            #             else:
            rule_items = self._decode(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
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
                if self._verbose:
                    print(self._verbose_header, "black box candid", bb_features[candid_feature])
                bb_features[candid_feature] += counter # IAIN this should be on the scale of the other rules or indicated as not used
                other_rules.append(rule)
            counter -= 1

        if self._verbose:
            print(self._verbose_header, "black box features", bb_features)
            print(self._verbose_header, "rules", applicable_rules,
                  applied_rules,
                  other_rules)
        feature_value_pairs = sorted(bb_features.items(), key=lambda x: x[1], reverse=True)

        return [(self._feature_names[k], v, "input comparison: " + str(np.round(list(data_row)[k], 4))) for k, v in feature_value_pairs]

    def get_translation(self):
        # IAIN we should use this to translate bin values given in the feature value pairs
        pass
