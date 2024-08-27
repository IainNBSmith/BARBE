"""
The point of this code is to handle sigdirect output in a way that would be improper to handle on
 the side of BARBE.

"""
import itertools
import os
from collections import defaultdict

import pandas as pd
# from distutils.core import setup
# from Cython.Build import cythonize

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer

# IAIN requires main folder to contain rule.py
from sigdirect import SigDirect
import numpy as np


class SigDirectWrapper:
    __doc__ = '''
        Purpose: Utility for BARBE, handles some operations that are not in SigDirect by default.

        Input: column_names (list<string>) -> Original column names of input data. Required to get named rules.
               verbose (boolean)           -> Whether to provide verbose output.
        '''

    def __init__(self, column_names, n_bins=5, verbose=False):
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
        self._n_bins = n_bins
        self._oh_enc = None
        self._kb_discrete = None
        self._rules = None
        self._feature_names = column_names
        self._categorical_features = None
        self._verbose = verbose
        self._verbose_header = "SigDirect:"

    def _create_encoder(self, X):
        if self._verbose:
            print(self._verbose_header, "on encoder creation:", X.shape, X)

        self._categorical_features = dict()
        for i in range(X.shape[1]):
            unique_features = np.unique(X[:, i])
            # check that there are little enough unique values to be considered discrete
            if len(unique_features) <= 10:
                self._categorical_features[i] = unique_features
            elif not np.isscalar(unique_features):
                assert ValueError(self._verbose_header + " ERROR: features with more than 10 distinct values must be "
                                                         "scalar.")
        self._not_categorical = [i for i in range(X.shape[1]) if i not in list(self._categorical_features.keys())]

        self._oh_enc = OneHotEncoder(categories='auto', handle_unknown='ignore',
                                     min_frequency=None)
        self._kb_discrete = KBinsDiscretizer(n_bins=self._n_bins, encode='ordinal', strategy='quantile')
        self._kb_discrete.fit(X[:, self._not_categorical])
        # two encoding modes, one with mixture of bins and discrete the other only has bins
        if list(self._categorical_features.keys()):
            self._oh_enc.fit(np.append(self._kb_discrete.transform(X[:, self._not_categorical]),
                                       self._categorical_transform(X), axis=1))
        else:
            self._oh_enc.fit(self._kb_discrete.transform(X))
        self._new_feature_order = self._not_categorical + list(self._categorical_features.keys())

    def _categorical_transform(self, X):
        X = X.copy()
        for index, uvalues in self._categorical_features.items():
            replacement_array = np.array([None for i in range(X.shape[0])])
            for i in range(len(uvalues)):
                replacement_array[np.where(X[:, index] == uvalues[i])] = i
            X[:, index] = replacement_array
        return X[:, list(self._categorical_features.keys())]

    def _encode(self, X):
        if self._verbose:
            print(self._verbose_header, "before encoding:", X.shape, X)
        if list(self._categorical_features.keys()):
            return np.asarray(self._oh_enc.transform(
                np.append(self._kb_discrete.transform(X[:, self._not_categorical]),
                          self._categorical_transform(X), axis=1)).todense()).astype(int)
        else:
            return np.asarray(self._oh_enc.transform(self._kb_discrete.transform(X)).todense()).astype(int)

    def _decode(self, enc_X):
        partial_decoding = self._oh_enc.inverse_transform(enc_X)[0]
        partial_usable = [i for i in range(len(partial_decoding)) if partial_decoding[i] is not None]
        if self._verbose:
            print(self._verbose_header, "before decoding:", enc_X)
            print(self._verbose_header, "OneHot decode:", partial_decoding)
            print(partial_usable, partial_decoding[partial_usable])
            #print(self._kb_discrete.bin_edges_[partial_usable][0])
            #print(self._verbose_header, "corresponding bins",
            #      self._kb_discrete.bin_edges_[partial_usable][0][partial_decoding[partial_usable].astype(int)])

        # do not need to fully decode I only need to handle which bin value is assigned
        return_list = [None for i in range(len(partial_decoding))]
        for i in partial_usable:
            moved_index = self._new_feature_order[i]
            if moved_index in self._not_categorical:
                converted_index = np.where(moved_index == np.array(self._not_categorical))[0][0]

                return_list[moved_index] = self._kb_discrete.bin_edges_[converted_index][int(partial_decoding[i])]
            else:
                converted_index = np.where(moved_index == np.array(list(self._categorical_features.keys())))[0][0]
                print(converted_index, i, np.array(list(self._categorical_features.keys())))
                print(self._categorical_features)
                # IAIN URGENT (CHECK IF THE MOVED INDEX IS CORRECT HERE)
                return_list[moved_index] = self._categorical_features[moved_index][int(partial_decoding[i])]

        return [return_list]

    def _y_conversion(self, y):
        y = y.tolist()
        self._pred_map = np.unique(y)
        return [np.where(self._pred_map == yi)[0][0] for yi in y]

    def fit(self, X, y):
        # create an encoder for the given values
        X = pd.DataFrame(X)
        X = X.fillna(0)
        X = X.to_numpy()
        y = self._y_conversion(y)
        print(X[0,:])
        self._create_encoder(X)
        if self._verbose:
            print(self._verbose_header, self._encode(X))
            print(self._verbose_header, "training y:", y)
        # train sigdirect on one hot encoding of input data [0, 1, 0, 1, 1, ...]
        print(self._encode(X))
        self._sigdirect_model.fit(self._encode(X), y)

    def _y_reconversion(self, y):
        return self._pred_map[y]

    def predict(self, X):
        # print("IAIN NEW: ", X[0,:])
        return self._y_reconversion(self._sigdirect_model.predict(self._encode(X)))

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
                all_rules[self._y_reconversion(x)] = [(t, self._oh_enc, self._encode(data_row.to_numpy().reshape(1, -1))) for t in y]

        else:
            if self._verbose:
                print(self._verbose_header, "rules did not match", predicted_label, true_label)
            predicted_label = -1  # to show we couldn't predict it correctly
        self._rules = all_rules
        return predicted_label

    def _get_bin_bounds(self, bin_index, current_item):
        bin_size = len(self._kb_discrete.bin_edges_[bin_index])
        item_index = np.where(self._kb_discrete.bin_edges_[bin_index] == current_item)[0]
        if item_index == 0:
            return None, current_item
        if item_index == bin_size - 2:
            return current_item, None
        return current_item, self._kb_discrete.bin_edges_[bin_index][item_index+1][0]

    def get_categories(self):
        return self._oh_enc.categories_

    def raw_rule_translation(self, raw_enc, impl_str):
        if raw_enc is None:
            return None
        # print('IAIN sig_side: ', self._decode(raw_enc.reshape((1,-1))))
        return self._rule_translation(self._decode(raw_enc.reshape((1,-1)))[0]) + " -> " + impl_str

    def _rule_translation(self, rule_items):
        rule_text = ""
        for i in range(len(rule_items)):
            item = rule_items[i]
            moved_index = i
            if item is not None:
                if rule_text != "":
                    rule_text += ", "
                if moved_index in self._not_categorical:
                    converted_index = np.where(moved_index == np.array(self._not_categorical))[0][0]
                    rule_low, rule_high = self._get_bin_bounds(converted_index, item)
                    if rule_low is None:
                        rule_text += ("'" + str(self._feature_names[i]) + "'" +
                                      " < " + str(np.round(rule_high, 4)))
                    elif rule_high is None:
                        rule_text += ("'" + str(self._feature_names[i]) + "'" +
                                      " > " + str(np.round(rule_low, 4)))
                    else:
                        rule_text += (str(np.round(rule_low, 4)) + " <= " +
                                      "'" + str(self._feature_names[i]) + "'" +
                                      " <= " + str(np.round(rule_high, 4)))
                else:
                    rule_text += ("'" + str(self._feature_names[moved_index]) + "'" + " = " + "'" + str(item) + "'")
        return rule_text

    def get_ohe_simple(self):
        """
        Input:
        Purpose: Return a simple vector that denotes the bin positions of the one hot encoder.
        Output:
        """
        #print("IAIN in ohe simple")
        # original_size = self._rules[0][0][2].shape[1]
        print(self._rules)
        for item in self._rules:
            original_size = self._rules[item][0][2].shape[1]
            break
        try:
            ohe_key = np.zeros(original_size).astype(int)
            prev_ind = None
            counter = 0
            for i in range(original_size):
                temp = np.zeros(original_size).astype(int)
                temp[i] = 1
                position = self._decode(temp.reshape((1, -1)))[0]
                # print("IAIN OHE POSITION: ", position)
                ind_use = np.where(np.array(position) != None)
                # print("IAIN OHE IND USE: ", ind_use)
                if prev_ind is None or prev_ind == ind_use:
                    counter += 1
                else:
                    counter = 1
                ohe_key[i] = counter
                prev_ind = ind_use
        except UnboundLocalError:
            assert ValueError(self._verbose_header + " ERROR: no rules available. Check input data and settings.")

        #print("IAIN SIMPLE OHE: ", ohe_key)
        return ohe_key

    def get_applicable_rules(self, input_point):
        rules_subset = self._rules
        input_enc = self._encode(input_point)[0]
        input_point = np.where(input_enc == 1)[0].tolist()

        rules_translation = []
        for class_label in rules_subset:
            for rule, _, original_point_sd in rules_subset[class_label]:
                # with one hot encoded vector only non zero values from the rule need to be set to 1
                # print("IAIN RULE ITEMS: ", rule.get_items())
                if all([ind in input_point for ind in rule.get_items()]):
                    temp = np.zeros(original_point_sd.shape[1]).astype(int)
                    temp[rule.get_items()] = 1
                    # get features from the rule to store
                    rule_support = rule.get_support()
                    rule_confidence = rule.get_confidence()
                    rule_p = rule.get_log_p()
                    # decode one hot vector and get text version of the rule
                    rule_items = self._decode(temp.reshape((1, -1)))[0]
                    rule_text = self._rule_translation(rule_items)
                    rules_translation.append((rule_text + " -> " + str(class_label), class_label,
                                              rule_support, rule_confidence, rule_p))
        # format [(rule text, support, confidence, rule_p), ...]
        return rules_translation



    def get_all_rules(self, rules_subset=None):
        if rules_subset is None:
            rules_subset = self._rules

        rules_translation = []
        for class_label in rules_subset:
            for rule, _, original_point_sd in rules_subset[class_label]:
                # with one hot encoded vector only non zero values from the rule need to be set to 1
                temp = np.zeros(original_point_sd.shape[1]).astype(int)
                temp[rule.get_items()] = 1
                # get features from the rule to store
                rule_support = rule.get_support()
                rule_confidence = rule.get_confidence()
                rule_p = rule.get_log_p()
                # decode one hot vector and get text version of the rule
                rule_items = self._decode(temp.reshape((1, -1)))[0]
                rule_text = self._rule_translation(rule_items)
                rules_translation.append((rule_text + " -> " + str(class_label), class_label,
                                          rule_support, rule_confidence, rule_p))
        # format [(rule text, support, confidence, rule_p), ...]
        return rules_translation

    def get_contrast_sets(self, data_row, max_dev=0.0005, raw_rules=False, new_class=0, old_class=1):
        # IAIN get contrast sets as in the paper "Learning Statistically Significant Contrast Sets" Algorithm 1
        # encode and decode the data_row (to get the applied bins)
        encoded_value = self._encode(data_row.to_numpy().reshape(1, -1))
        len_encoding = len(encoded_value[0])
        data_antecedent = self._decode(encoded_value)[0]
        print(data_antecedent)

        # Get the kingfisher association rules
        label_quality = self._generate_rules(data_row, self.predict(data_row.to_numpy().reshape(1, -1)))
        all_rules = self._rules

        # translate the rules into values
        all_translated_rules = dict()
        for c_label in all_rules:
            rule_list = all_rules[c_label]
            all_translated_rules[c_label] = []
            for rule, _, _ in rule_list:
                temp = np.zeros(len_encoding).astype(int)
                temp[rule.get_items()] = 1
                all_translated_rules[c_label].append((self._decode(temp.reshape(1, -1))[0], temp.reshape(1, -1),
                                                      rule.get_confidence(), 10**rule.get_log_p()))

        #print("IAIN ALL RULES ", all_translated_rules)

        final_rules = []
        # for each rule check that it satisfies eq 3 not all classes are the same for the condition p-value
        for c_label, rule_list in all_translated_rules.items():
            for rule, raw, conf, p_val in rule_list:
                min_dif = abs(p_val - 0.0005)
                non_equal = False
                e_match = 0
                for o_c_label, o_rule_list in all_translated_rules.items():
                    if ((c_label == new_class and o_c_label == old_class) or
                            (c_label == old_class and o_c_label == new_class)):
                        for o_rule, _, _, o_p_val in o_rule_list:
                            # IAIN or set to equal in other cases
                            if all([(rule[i] is not None and o_rule[i] is not None) or
                                    (rule[i] is None and o_rule[i] is None)
                                    for i in range(len(rule))]):
                                e_match += 1
                                if p_val != o_p_val:
                                    #print("IAIN POTENTIAL PAIR ", rule, o_rule)
                                    non_equal = True
                                    if abs(p_val - o_p_val) < min_dif or min_dif == abs(p_val) - 0.05:
                                        min_dif = abs(p_val - o_p_val)

                #print("IAIN WHY NOT ", min_dif, max_dev)
                if min_dif <= max_dev and not (e_match != 0 and not non_equal):
                    #print("IAIN ADDED RULE")
                    # for each significant rule if c.antecedant < o.antecedent (X -> c, X is antecedent, o=data_row)
                    #  add it to set
                    # these now contain the contrast sets
                    # return the rules (in text) that apply to o (data_row)
                    # IAIN remove line for just rules that apply to current if all([(rule[i] == data_antecedent[i]) or
                    #  (rule[i] is None) for i in range(len(data_antecedent))]):
                    # IAIN we could even append a set of rules that are together considered (applied + counter)
                    if raw_rules:
                        final_rules.append((raw[0], c_label, conf, p_val))
                    else:
                        final_rules.append((rule, c_label, p_val, min_dif))

        if raw_rules:
            #print("IAIN FINAL RULES ", final_rules)
            return final_rules
        return [(self._rule_translation(rule_item) + ' -> ' + str(c), c, p, m) for rule_item, c, p, m in final_rules]

    def get_features(self, data_row, true_label):
        # IAIN now one thing to try is when decoding we just check where relevant bins to the sample appear,
        #  I'm pretty sure we would still do decoding for this though
        # print("Bins:", self._kb_discrete.bin_edges_)
        label_quality = self._generate_rules(data_row, true_label)
        if label_quality == -1:
            return None
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
        bb_features = defaultdict(lambda: [])

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
                bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(), 1))
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
                    bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(), counter))
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
                # bb_features[candid_feature] += counter # IAIN this should be on the scale of the other rules or
                #  indicated as not used
                bb_features[candid_feature].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(), counter))
                other_rules.append(rule)
            # what if we subtract the rule's support instead
            counter -= 1

        if self._verbose:
            print(self._verbose_header, "black box features", bb_features)
            print(self._verbose_header, "rules", applicable_rules,
                  applied_rules,
                  other_rules)

        bb_features = evaluation_function(bb_features)
        feature_value_pairs = sorted(bb_features.items(), key=lambda x: x[1], reverse=True)

        return [(self._feature_names[k], v) for k, v in feature_value_pairs]


def evaluation_function(bb_features):
    # expect dict<feature, list<(support, confidence, log_p, counter)>]>
    # return dict<feature, float>
    eval_bb_features = defaultdict(int)
    SMALL_FLOAT = 1e-9
    high_importance = 0
    for feature, tlist in bb_features.items():
        temp_eval = 0
        for support, confidence, pvalue, counter in tlist:
            if counter > 0 or feature not in eval_bb_features.keys():
                temp_eval += np.sign(counter) * ( (1/(support+SMALL_FLOAT)) * (1/(confidence+SMALL_FLOAT)) *
                                                  (1/((10**pvalue)+SMALL_FLOAT)) )
        temp_eval /= len(tlist)
        eval_bb_features[feature] = temp_eval
        if temp_eval > high_importance:
            high_importance = temp_eval
    for feature in eval_bb_features.keys():
        eval_bb_features[feature] /= high_importance
    return eval_bb_features