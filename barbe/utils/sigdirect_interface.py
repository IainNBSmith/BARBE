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
#from sklearn.decomposition import PCA
import numpy as np


class SigDirectWrapper:
    __doc__ = '''
        Purpose: Utility for BARBE, handles some operations that are not in SigDirect by default.

        Input: column_names (list<string>) -> Original column names of input data. Required to get named rules.
               verbose (boolean)           -> Whether to provide verbose output.
        '''

    def __init__(self, column_names, n_bins=5, use_negative_rules=True, verbose=False):
        # IAIN this should have settings for sigdirect and more accurate utilities
        # IAIN intention is for the user to get the same flexibility as scikit
        # IAIN adjusting settings worked!!
        # IAIN confidence=0.5 and alpha=0.001 and early_stopping=True gets 0.2222
        self._sigdirect_model = SigDirect(
                clf_version=2,  # 1 is without better trimming version 2 is SigD2
                alpha=0.001,  #0.01 may be ok TODO: try alpha values for inverse...
                early_stopping=False,
                confidence_threshold=0.5,  # was 0.5 TODO: try changing for the inverse rules
                is_binary=True,
                get_logs=False,
                other_info=None)
        #self._sigdirect_model = SigDirect(
        #    clf_version=1,
        #    alpha=0.1,
        #    early_stopping=False,
        #    confidence_threshold=0.2,
        #    is_binary=True,
        #    get_logs=False,
        #    other_info=None)
        self._use_negative_rules = use_negative_rules
        self._n_bins = n_bins
        self._oh_enc = None
        self._kb_discrete = None
        self._rules = None
        self._feature_names = column_names.copy()
        self._categorical_features = None
        self._verbose = verbose
        self._verbose_header = "SigDirect:"
        self._n_encoded_features = None


    def _create_encoder(self, X):
        if self._verbose:
            print(self._verbose_header, "on encoder creation:", X.shape, X)

        self._categorical_features = dict()
        for i in range(X.shape[1]):
            unique_features = np.unique(X[:, i])
            # check that there are little enough unique values to be considered discrete
            #print(unique_features)
            #print(len(unique_features))
            if len(unique_features) <= 20:
                self._categorical_features[i] = unique_features
            elif not np.isscalar(unique_features):
                assert ValueError(self._verbose_header + " ERROR: features with more than 10 distinct values must be "
                                                         "scalar.")
        self._not_categorical = [i for i in range(X.shape[1]) if i not in list(self._categorical_features.keys())]

        #print(self._categorical_features)
        #print(self._not_categorical)
        #self._pca = PCA(n_components=X.shape[1] if X.shape[1] <= 30 else 30)
        self._oh_enc = OneHotEncoder(categories='auto', handle_unknown='ignore',
                                     min_frequency=None)
        self._kb_discrete = KBinsDiscretizer(n_bins=self._n_bins, encode='ordinal', strategy='quantile')
        #print(self._not_categorical)
        #print(X)
        #assert False
        #print(self._categorical_features)
        self._kb_discrete.fit(X[:, self._not_categorical])
        # two encoding modes, one with mixture of bins and discrete the other only has bins
        if list(self._categorical_features.keys()):
            self._oh_enc.fit(np.append(self._kb_discrete.transform(X[:, self._not_categorical]),
                                       X[:, list(self._categorical_features.keys())], axis=1))
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
            ret_X = np.asarray(self._oh_enc.transform(
                np.append(self._kb_discrete.transform(X[:, self._not_categorical]),
                          X[:, list(self._categorical_features.keys())], axis=1)).todense()).astype(int)
        else:
            ret_X = np.asarray(self._oh_enc.transform(self._kb_discrete.transform(X)).todense()).astype(int)

        # IAIN NEW
        if self._n_encoded_features is None:
            self._n_encoded_features = ret_X.shape[1]

        if self._use_negative_rules:
            #print(ret_X.shape)
            ret_neg_X = np.abs(ret_X - 1)
            ret_X = np.concatenate([ret_X, ret_neg_X], axis=1)
            #print("ENCODED: ", ret_X[0, :])

        return ret_X

    def _decode(self, enc_X, as_pd=False):
        # NEW IAIN
        if self._use_negative_rules:
            enc_X = enc_X[:, 0:self._n_encoded_features]

        if self._verbose:
            print(self._verbose_header, "encoded after fixing", enc_X.shape, enc_X)



        part_X = self._oh_enc.inverse_transform(enc_X.copy())
        if self._verbose:
            print(self._verbose_header, "partially decoded part", part_X)

        if list(self._categorical_features.keys()):
            #ret_X = np.asarray(self._oh_enc.transform(
            #    np.append((X[:, self._not_categorical]),
            #              X[:, list(self._categorical_features.keys())], axis=1)).todense()).astype(int)
            categorical_feature_len = len(list(self._categorical_features.keys()))
            numeric_part = np.array(part_X[:, :(part_X.shape[1] - categorical_feature_len)], dtype=float)
            dec_numeric = self._kb_discrete.inverse_transform(np.nan_to_num(numeric_part))
            dec_numeric[np.isnan(numeric_part)] = None
            dec_X = np.append(dec_numeric,
                              part_X[:, (part_X.shape[1] - categorical_feature_len):], axis=1)

        else:
            part_X = np.array(part_X, dtype=float)
            dec_X = self._kb_discrete.inverse_transform(np.nan_to_num(part_X))
            dec_X[pd.isna(part_X)] = None

        # return them to their original order
        order = np.argsort(self._new_feature_order)
        #print(self._verbose_header, 'final ordered', order, dec_X, dec_X[:, order])
        if as_pd:
            dec_X = pd.DataFrame(dec_X[:, order], columns=self._feature_names)
            return dec_X
        else:
            return dec_X[:, order]


    def _old_decode(self, enc_X):
        # TODO: IAIN we need to find what values were affected by the PCA and put those into the rule instead
        partial_decoding = self._oh_enc.inverse_transform(enc_X)[0]

        # TODO: IAIN need to fix decoding
        #print("IAIN PARTIAL: ", partial_decoding)
        partial_usable = [i for i in range(len(partial_decoding)) if partial_decoding[i] is not None]
        #partial_decoding[[True if temp is None else False for temp in partial_decoding]] =
        #self._pca.inverse_transform(
        if self._verbose:
            print(self._verbose_header, "before decoding:", enc_X)
            print(self._verbose_header, "OneHot decode:", partial_decoding)
            print(partial_usable, partial_decoding[partial_usable])

        # do not need to fully decode I only need to handle which bin value is assigned
        return_list = [None for i in range(len(partial_decoding))]
        for i in partial_usable:
            moved_index = self._new_feature_order[i]
            if moved_index in self._not_categorical:
                converted_index = np.where(moved_index == np.array(self._not_categorical))[0][0]
                bin_pos = (int(partial_decoding[i]) + 2) if int(partial_decoding[i]) == 0 else (
                        int(partial_decoding[i]) + 1)
                bin_min = self._kb_discrete.bin_edges_[converted_index][bin_pos-1]
                bin_max = self._kb_discrete.bin_edges_[converted_index][bin_pos]
                #print("BIN EDGES: ", self._kb_discrete.bin_edges_[converted_index])
                #assert False
                return_list[moved_index] = (bin_max + bin_min)/2
            else:
                converted_index = np.where(moved_index == np.array(list(self._categorical_features.keys())))[0][0]
                #print(converted_index, i, np.array(list(self._categorical_features.keys())))
                #print(self._categorical_features)
                # IAIN URGENT (CHECK IF THE MOVED INDEX IS CORRECT HERE)
                #print(return_list)
                return_list[moved_index] = partial_decoding[i]

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
        #print(X[0,:])
        self._create_encoder(X)
        if self._verbose:
            print(self._verbose_header, self._encode(X))
            print(self._verbose_header, "training y:", y)
        # train sigdirect on one hot encoding of input data [0, 1, 0, 1, 1, ...]
        #print(self._encode(X))
        self._sigdirect_model.fit(self._encode(X), y)

    def _y_reconversion(self, y):
        return self._pred_map[y]

    def predict(self, X):
        # print("IAIN NEW: ", X)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self._y_reconversion(self._sigdirect_model.predict(self._encode(X), hrs=3))  # set heuristic to 1

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
        # called to get the bounds then do other ops
        # bin index is acquired elsewhere
        bin_size = len(self._kb_discrete.bin_edges_[bin_index])
        # find the first bound where the
        #print(self._kb_discrete.bin_edges_)
        #print(self._kb_discrete.bin_edges_[bin_index])
        #print(bin_index)
        #print(current_item)
        item_index = np.min(np.argwhere(self._kb_discrete.bin_edges_[bin_index] >= current_item))
        #print("GOD WHY OH GOD: ", item_index)
        if item_index == 1:
            return None, self._kb_discrete.bin_edges_[bin_index][item_index-1]
        if item_index == bin_size - 1:
            return current_item, None
        # TODO: may need to swap back
        return self._kb_discrete.bin_edges_[bin_index][item_index-1], current_item

    def get_categories(self):
        return self._oh_enc.categories_

    def raw_rule_translation(self, raw_enc, impl_str):
        if raw_enc is None:
            return None
        # print('IAIN sig_side: ', self._decode(raw_enc.reshape((1,-1))))
        return self._rule_translation(raw_enc) + " -> " + impl_str

    def get_bounded_translation(self, enc_value):
        translated_value = dict()
        for i in range(len(enc_value)):
            item = enc_value[i]
            moved_index = i
            if item is not None:
                if moved_index in self._not_categorical:
                    converted_index = np.where(moved_index == np.array(self._not_categorical))[0][0]
                    translated_value[str(self._feature_names[i])] = self._get_bin_bounds(converted_index, item)
                else:
                    translated_value[str(self._feature_names[moved_index])] = str(item)

        return translated_value

    def _rule_translation(self, rule_vector):
        rule_text = ""
        for i in range(len(rule_vector)):
            if rule_vector[i] == 1:
                is_negation = self._use_negative_rules and i >= self._n_encoded_features
                ii = i if i < self._n_encoded_features else i - self._n_encoded_features
                temp = np.zeros(rule_vector.shape)
                temp[ii] = 1
                #print(self._verbose_header, 'zeroes to decode', temp.reshape((1, -1)))
                rule_items = self._decode(temp.reshape((1, -1)))[0]
                #print(self._verbose_header, 'decoded items for rules', rule_items)
                ii = np.argwhere(~pd.isna(rule_items))[0][0]
                item = rule_items[ii]
                moved_index = ii
                if item is not None:
                    if rule_text != "":
                        rule_text += ", "
                    if is_negation:
                        rule_text += '~('
                    if moved_index in self._not_categorical:
                        converted_index = np.where(moved_index == np.array(self._not_categorical))[0][0]
                        rule_low, rule_high = self._get_bin_bounds(converted_index, item)
                        if rule_low is None:
                            rule_text += ("'" + str(self._feature_names[ii]) + "'" +
                                          " < " + str(np.round(rule_high, 4)))
                        elif rule_high is None:
                            rule_text += ("'" + str(self._feature_names[ii]) + "'" +
                                          " > " + str(np.round(rule_low, 4)))
                        else:
                            rule_text += (str(np.round(rule_low, 4)) + " <= " +
                                          "'" + str(self._feature_names[ii]) + "'" +
                                          " <= " + str(np.round(rule_high, 4)))
                    else:
                        rule_text += ("'" + str(self._feature_names[moved_index]) + "'" + " = " + "'" + str(item) + "'")

                    if is_negation:
                        rule_text += ')'
        return rule_text

    def get_ohe_simple(self):
        """
        Input:
        Purpose: Return a simple vector that denotes the bin positions of the one hot encoder.
        Output:
        """
        #print("IAIN in ohe simple")
        # original_size = self._rules[0][0][2].shape[1]
        #print(self._rules)
        for item in self._rules:
            #print(item)
            try:
                original_size = self._rules[item][0][2].shape[1]
                break
            except:
                print(self._rules[item])
        try:
            ohe_key = np.zeros(original_size).astype(int)
            prev_ind = None
            counter = 0
            for i in range(original_size):
                temp = np.zeros(original_size).astype(int)
                temp[i] = 1
                position = self._decode(temp.reshape((1, -1)))[0]
                #print("IAIN OHE POSITION: ", position)
                ind_use = np.where(np.array(position) != None)
                # print("IAIN OHE IND USE: ", ind_use)
                if prev_ind is None or prev_ind == ind_use:
                    counter += 1
                else:
                    counter = 1
                ohe_key[i] = counter
                prev_ind = ind_use
            return ohe_key
        except:
            raise ValueError(self._verbose_header + " ERROR: no rules available. Check input data and settings.")

        #print("IAIN SIMPLE OHE: ", ohe_key)
        return None

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
                    rule_text = self._rule_translation(temp)
                    rules_translation.append((rule_text + " -> " + str(class_label), class_label,
                                              rule_support, rule_confidence, rule_p))
        # format [(rule text, support, confidence, rule_p), ...]
        return rules_translation

    def get_negation_index(self):
        return self._n_encoded_features

    def get_all_rules(self, rules_subset=None, raw_rules=False):
        if rules_subset is None:
            rules_subset = self._rules

        rules_translation = []
        for class_label in rules_subset:
            for rule, _, original_point_sd in rules_subset[class_label]:
                # with one hot encoded vector only non zero values from the rule need to be set to 1
                temp = np.zeros(original_point_sd.shape[1]).astype(int)
                #print(self._verbose_header, 'rule items', rule.get_items())
                temp[rule.get_items()] = 1
                # get features from the rule to store
                rule_support = rule.get_support()
                rule_confidence = rule.get_confidence()
                rule_p = rule.get_log_p()
                # decode one hot vector and get text version of the rule
                rule_items = self._decode(temp.reshape((1, -1)))[0]
                if raw_rules:
                    rules_translation.append((temp.reshape((1, -1))[0],
                                              class_label,
                                              rule_support, rule_confidence, rule_p))
                else:
                    rule_text = self._rule_translation(temp)
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
        #print(data_antecedent)

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
                    #if raw_rules:
                    final_rules.append((raw[0], c_label, conf, p_val))
                    #else:
                    #    final_rules.append((rule, c_label, p_val, min_dif))

        if raw_rules:
            #print("IAIN FINAL RULES ", final_rules)
            return final_rules
        return [(self._rule_translation(rule_item) + ' -> ' + str(c), c, p, m) for rule_item, c, p, m in final_rules]

    def get_features(self, data_row, true_label, next_best_class):
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
        applicable_sorted_rules = sorted(itertools.chain(*[all_rules[x] for x in all_rules if x == next_best_class]),
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

            # IAIN that one is my fault had not all(...)
            #if all([(temp[i] == original_point_sd[0][i]) or (temp[i] == 0) for i in range(len(temp))]):
            applicability = sum([(temp[i] == original_point_sd[0][i]) or (temp[i] == 0) for i in range(len(temp))])
            useful_applicability = sum([(temp[i] == original_point_sd[0][i]) and (original_point_sd[0][i] != 0) for i in range(len(temp))])
            #print(temp)
            #print(original_point_sd)
            #print([(temp[i] == original_point_sd[0][i]) or (temp[i] == 0) for i in range(len(temp))])
            #print(applicability)
            #print(len(temp))
            #assert False
            applied_rules.append(rule)
            influence = applicability / len(temp)
            #if applicability != len(temp):
            #    influence = 0
            # rule_items = ohe.inverse_transform(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
            rule_items = self._decode(temp.reshape((1, -1)))[0]
            rule_applicable = self._decode(np.array([(temp[i] == original_point_sd[0][i]) and (temp[i] != 0) for i in range(len(temp))]).reshape((1, -1)))[0]
            #         rule_items = temp ## TEXT (uncomment for TEXT)
            rule_items = [item for item in enumerate(rule_items)]
            rule_applicable = [item for item in enumerate(rule_applicable)]
            for i in range(len(rule_items)):
                item, val = rule_items[i]
                citem, cval = rule_applicable[i]
                if val is not None:
                    equiv_check = cval is not None
                    #print("IAIN HERE: ")
                    #print(item, citem, val, cval)
                    influence_sign = 1 if item == citem and val != cval else 1
                    # to change back remove the temp + 2 to only temp
                    influence_modifier = influence_sign*((1+2*useful_applicability) / (2*sum(temp)+2))
                    # FOR OLD
                    #bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(),
                    #                          sum(temp), useful_applicability))
                    bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(),
                                                  sum(temp), useful_applicability, True, equiv_check))
            #                     bb_features[item] += counter
            #                 bb_features[item] = max(bb_features[item],  rule.get_confidence()/len(rule.get_items()))
           # counter -= 1
        set_size_1 = len(bb_features)

        # Second, add applicable rules
        if True:
            applicable_rules = []
            for rule, ohe, original_point_sd in applicable_sorted_rules:
                temp = np.zeros(original_point_sd.shape[1]).astype(int)
                temp[rule.get_items()] = 1
                if self._verbose:
                    print(self._verbose_header, "item of note in applicable rules", rule.get_items())
                    print(self._verbose_header, "encoding meaning", self._decode(temp.reshape((1, -1))))

                #if all([(temp[i] == original_point_sd[0][i]) or (temp[i] == 0) for i in range(len(temp))]):
                applicability = sum(
                    [(temp[i] == original_point_sd[0][i]) or (temp[i] == 0) for i in range(len(temp))])
                useful_applicability = sum(
                    [(temp[i] == original_point_sd[0][i]) and (original_point_sd[0][i] != 0) for i in range(len(temp))])

                influence = - 1
                #if applicability != len(temp):
                #    influence = 0
                applicable_rules.append(rule)
                rule_items = self._decode(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
                rule_applicable = self._decode(
                    np.array([(temp[i] == original_point_sd[0][i]) and (temp[i] != 0) for i in range(len(temp))]).reshape((1, -1)))[0]
                #         rule_items = temp ## TEXT (uncomment for TEXT)
                rule_items = [item for item in enumerate(rule_items)]
                rule_applicable = [item for item in enumerate(rule_applicable)]
                for i in range(len(rule_items)):
                    item, val = rule_items[i]
                    citem, cval = rule_applicable[i]
                    #print("Item Val Pair: ", item, val)
                    if val is not None:
                        equiv_check = cval is not None
                        #                 bb_features[item] += rule.get_support()
                        influence_sign = -1 if item == citem and val != cval else 1
                        # influence_modifier = influence_sign*((1+2*useful_applicability) / (2*sum(temp)+2))
                        # FOR OLD
                        #bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(),
                        #                          sum(temp), useful_applicability))
                        bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(),
                                                  sum(temp), useful_applicability, False, equiv_check))
                #counter -= 1

        # Third, add other rules.
        # IAIN try every rule and make rules less similar to the current value the least important
        if False:
            other_rules = []
            for rule, ohe, original_point_sd in other_sorted_rules:
                temp = np.zeros(original_point_sd.shape[1]).astype(int)
                # tell you where the ohe is encoded
                temp[rule.get_items()] = 1
                if self._verbose:
                    print(self._verbose_header, "item of note in other", rule.get_items())
                    print(self._verbose_header, "encoding meaning", self._decode(temp.reshape((1, -1))))
                # avoid applicable rules
                #if np.array_equal(temp, temp & original_point_sd.astype(int)):  # error??? it was orig...[0].astype
                #    continue
                #             elif temp.sum()==1:
                #                 continue
                #elif temp.sum() - np.sum(temp & original_point_sd.astype(int)) > 1:  # error???
                #    continue
                #             else:
                rule_items = self._decode(temp.reshape((1, -1)))[0]  ## TEXT (comment for TEXT)
                #         rule_items = temp ## TEXT (uncomment for TEXT)
                n_matched_features = sum([(temp[i] == original_point_sd[0][i]) or (temp[i] == 0) for i in range(len(temp))])
                seen_set = 0
                #counter = 1 if rule.ge
                for item, val in enumerate(rule_items):
                    if val is not None:
                        bb_features[item].append((rule.get_support(), rule.get_confidence(), rule.get_log_p(), sum(temp), counter))
                        #other_rules.append(rule)
                #if seen_set == temp.sum() - 1:  # and (item not in bb_features):
                #    if self._verbose:
                #        print(self._verbose_header, "black box candid", bb_features[candid_feature])
                    # bb_features[candid_feature] += counter # IAIN this should be on the scale of the other rules or
                    #  indicated as not used

                # what if we subtract the rule's support instead
                counter -= 1

        #if self._verbose:
        #    print(self._verbose_header, "black box features", bb_features)
        #    print(self._verbose_header, "rules", applicable_rules,
        #          applied_rules,
        #          other_rules)

        bb_features = old_evaluation_function(bb_features)
        feature_value_pairs = sorted(bb_features.items(), key=lambda x: abs(x[1]), reverse=True)

        return [(self._feature_names[k], v) for k, v in feature_value_pairs]


def old_evaluation_function(bb_features):
    # expect dict<feature, list<(support, confidence, log_p, counter)>]>
    # return dict<feature, float>
    eval_bb_features = defaultdict(int)
    for feature, tlist in bb_features.items():
        temp_eval = 0
        n_appearances = 0
        pos_rules = 0
        for support, confidence, pvalue, n_rules, counter, class_val, fulfill in tlist:
            if n_rules == 1 or n_rules/2 <= counter:
                pos_rules += 1 if (class_val and fulfill) or (not class_val and not fulfill) else 0
                # pos_rules += 1 if n_rules == counter and class_val else 0
                #sign = 1 if class_val else -1
                #sign = -sign if n_rules != counter and not class_val else sign
                sign = 1
                adj_p = -pvalue if -pvalue <= 100 else 100
                temp_eval += sign*support#*confidence*adj_p
                n_appearances += 1
        if n_appearances != 0:
            eval_bb_features[feature] = temp_eval #/ n_appearances
        else:
            eval_bb_features[feature] = temp_eval
        #if pos_rules < n_appearances - pos_rules:
        #    eval_bb_features[feature] = -eval_bb_features[feature]
    return eval_bb_features


def new_evaluation_function(bb_features):
    # expect dict<feature, list<(support, confidence, log_p, counter)>]>
    # return dict<feature, float>
    eval_bb_features = defaultdict(int)
    SMALL_FLOAT = 1e-9
    high_importance = 0
    highest_positive_rules = 0
    for feature, tlist in bb_features.items():
        temp_eval = 0
        temp_positive_rules = 0
        temp_points_to = 0
        rule_satisfied = False
        # rule.get_support(), rule.get_confidence(), rule.get_log_p(),
        # sum(temp), useful_applicability, False, item == citem and val != cval)
        for support, confidence, logp, rule_len, n_satisfied, isc, ist in tlist:
            if n_satisfied == rule_len:
                rule_satisfied = True
                dynamic_weight = ((2*n_satisfied + 1)) / ((2*rule_len + 2))
                #dynamic_weight = 1 - dynamic_weight if rule_len == n_satisfied or not ist else dynamic_weight
                dynamic_weight = dynamic_weight if isc else -dynamic_weight

                adj_pval = abs(logp) if abs(logp) <= 100 else 100
                temp_eval += dynamic_weight * adj_pval

                temp_positive_rules += dynamic_weight
                #  if adj_counter != 0:
                if dynamic_weight != 0:
                    temp_points_to += 1
        if not rule_satisfied:
            for support, confidence, logp, rule_len, n_satisfied, isc, ist in tlist:
                if n_satisfied >= rule_len/2 or rule_len == 1:
                    dynamic_weight = ((2*n_satisfied + 1)) / ((2*rule_len + 2))
                    dynamic_weight = dynamic_weight if (isc and ist) or (not isc and ist) else -dynamic_weight

                    adj_pval = abs(logp) if abs(logp) <= 100 else 100
                    temp_eval += dynamic_weight * adj_pval

                    temp_positive_rules += dynamic_weight
                    #  if adj_counter != 0:
                    if dynamic_weight != 0:
                        temp_points_to += 1

        if temp_positive_rules != 0:
            # if swapping back then remove the absolute and absolute sorting
            temp_eval /= abs(temp_positive_rules)
        eval_bb_features[feature] = temp_eval  #temp_points_to*
        highest_positive_rules = temp_points_to if temp_points_to > highest_positive_rules else highest_positive_rules
        #if temp_eval > high_importance:
        #    high_importance = temp_eval
    #if highest_positive_rules > 0:
    #    for feature in eval_bb_features.keys():
    #        eval_bb_features[feature] /= highest_positive_rules
    return eval_bb_features


def ot_new_evaluation_function(bb_features):
    # expect dict<feature, list<(support, confidence, log_p, counter)>]>
    # return dict<feature, float>
    eval_bb_features = defaultdict(int)
    SMALL_FLOAT = 1e-9
    high_importance = 0
    highest_positive_rules = 0
    for feature, tlist in bb_features.items():
        temp_eval = 0
        temp_positive_rules = 0
        temp_points_to = 1
        # rule.get_support(), rule.get_confidence(), rule.get_log_p(),
        # sum(temp), useful_applicability, False, item == citem and val != cval)
        for support, confidence, logp, rule_len, n_satisfied, isc, ist in tlist:
            if n_satisfied >= rule_len/2 or rule_len == 1:
                dynamic_weight = ((2*n_satisfied + 1)) / ((2*rule_len + 2))  # (n_satisfied + 1)  #
                dynamic_weight = dynamic_weight if (isc and ist) or (not isc and not ist) else -dynamic_weight
                # count_hurts = 1/(2**temp_points_to) if n_satisfied != rule_len else 2  # make 2 if not good

                adj_pval = abs(logp) if abs(logp) <= 100 else 100
                temp_eval += dynamic_weight * adj_pval * support * confidence

                temp_positive_rules += dynamic_weight * support * confidence  # removed count_hurts from div term
                #  if adj_counter != 0:ope
                if dynamic_weight != 0 and n_satisfied != rule_len:
                    temp_points_to += 1

        if temp_positive_rules != 0:
            # if swapping back then remove the absolute and absolute sorting
            temp_eval /= abs(temp_positive_rules)
        eval_bb_features[feature] = temp_eval*temp_points_to
        highest_positive_rules = temp_points_to if temp_points_to > highest_positive_rules else highest_positive_rules
        #if temp_eval > high_importance:
        #    high_importance = temp_eval
    if highest_positive_rules > 0:
        for feature in eval_bb_features.keys():
            eval_bb_features[feature] /= highest_positive_rules
    return eval_bb_features
