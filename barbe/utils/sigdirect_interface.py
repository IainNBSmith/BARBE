"""
The point of this code is to handle sigdirect output in a way that would be improper to handle on
 the side of BARBE.
"""
import itertools
from collections import defaultdict

from sigdirect import SigDirect
import numpy as np


class SigDirectWrapper:
    def __init__(self):
        # IAIN this should have settings for sigdirect and more accurate utilities
        self._sigdirect_model = None
        pass

    def fit(self, X, y):
        # IAIN exactly as you would expect from sklearn
        pass

    def get_rules(self, neighborhood_data, labels_column, clf):
        # IAIN as originally written
        print('CALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
        clf.fit(neighborhood_data, labels_column)
        local_pred = clf.predict(neighborhood_data[0].reshape((1, -1)), 2).astype(int)[0]

        all_rules = clf.get_all_rules()
        return all_rules, local_pred

    def get_features(self, all_rules, true_label):
        # IAIN as originally written
        """
        Input: all_rules ()  ->
               true_label () ->
        Purpose: use applied rules first, and then the rest of the applicable rules, and then all rules (other labels,
         rest of them match)
        Output: feature_value_pairs () ->
        """

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

        feature_value_pairs = sorted(bb_features.items(), key=lambda x: x[1], reverse=True)

        return feature_value_pairs, None

    def get_translation(self):
        pass
