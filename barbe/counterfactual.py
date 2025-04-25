import numpy as np
import re
from collections import defaultdict

# TODO: add distance information to counterfactual
# TODO: check why even if you pass the current class it will still suggest a change (maybe I am too aggressive with changes?)
# TODO: Add input for rule changes that are allowed dict{feat_1:0, ...} 0 -> no changes, 1 -> + only, 2 -> - only, 3 -> +/-
#  TODO: we do this cause then if dict[feat1]: make change (then more checking for specifics)
# TODO: include option setting for how intense rule changes should be (e.g. only contrast or further...)
# TODO: Print the applicable rules and all rules into another file that we may review
# TODO: For the counterfactuals print the applicable rules into another file to to see why the prediction was made in one way
# TODO: Clean up test code so everything is clear
# TODO: Start implementing the test/trial for number classification comparison with DiCE


# TODO BIG: rewrite counterfactuals to use the raw rules


class OldBarbeCounterfactual:
    __doc__ = '''
        Purpose: Suggests modification of input data to change classification. Used and managed by BARBE explainer.

        Input: simple_distance (boolean) -> whether to use only simple distance comparison (no scale used).
                | Default: True
            '''

    def __init__(self, simple_distance=True):
        self._simple_distance = simple_distance

        self._distance_matrix = None
        # vector with importance values based on p-value * confidence
        self._importance_vector = None
        self._all_rules = None
        # key in the form [1, 2, 3, 1, 2, 1, 1, 1, ...]
        #                  binned         category
        self._position_key = None

    def _calculate_distance_matrix(self):
        self._distance_matrix = self._position_key
        j = 0
        for i in range(len(self._position_key)):
            if i + 1 < len(self._position_key) and self._position_key[i] <= self._position_key[i+1]:
                self._position_key[i] = j
            else:
                self._position_key[i] = j
                j += 1

        self._position_key[-1] = j-1

    def _calculate_importance_vector(self):
        if self._simple_distance:
            self._importance_vector = [1 for _ in range(len(self._position_key))]
        else:
            pass

    def fit(self, rules, position_key):
        """
        Input: rules (list<(enc_vec, cls, log(pval), 0)>) -> rules learned by BARBE.
               position_key (list<+int>)                      -> keys that show which feature each encoded value is.

        Purpose:

        Output:
        """
        self._all_rules = sorted(rules, key=lambda x: abs(x[2]))
        self._position_key = position_key

        self._calculate_distance_matrix()
        self._calculate_importance_vector()

    def _get_applicable_rules(self, data_enc, data_cls):
        # IAIN get all applicable rules for the given row (what to change)
        # rule<(rule vec, class, conf, supp, pval)>
        # for rule_vec, rule_cls, rule_cnf, rule_sup, rule_pv in self._all_rules:
        applicable_rules = []
        #print("IAIN ALL RULES: ", self._all_rules)
        for i in range(len(self._all_rules)):
            #print(self._all_rules[i])
            rule_vec, rule_cls, _, _ = self._all_rules[i]
            #print("IAIN CLASSES: ", rule_cls, data_cls)
            if rule_cls == data_cls:
                #print("IAIN FOUND POTENTIAL RULE")
                #print([(rule_vec[j] == data_row[j] or (rule_vec[j] == 0 and self._distance_matrix[j] != 1)) for j in range(len(data_row))])
                #print(rule_vec)
                #print(data_row)

                if np.array([(rule_vec[j] == data_enc[j] or (rule_vec[j] == 0)) for j in range(len(data_row))]).all():
                    applicable_rules.append(self._all_rules[i])

        return applicable_rules

    def _all_same_features(self, row1, row2):
        #print("IAIN KEY: ", self._position_key)
        #print("IAIN CHECK1: ", row1)
        #print("IAIN CHECK2: ", row2)
        loaded_feature = None
        for i in range(len(row1)):
            if loaded_feature is None and (row1[i] == 1 or row2[i] == 1) and not (row1[i] == 1 and row2[i] == 1):
                loaded_feature = self._position_key[i]
            elif loaded_feature is not None and (row1[i] == 1 or row2[i] == 1) and not (row1[i] == 1 and row2[i] == 1):
                if loaded_feature != self._position_key[i]:
                    return False
                loaded_feature = None
        return True

    def _count_same_features(self, row1, row2):
        loaded_feature = None
        count = 0
        for i in range(len(row1)):
            if loaded_feature is None and (row1[i] == 1 or row2[i] == 1) and not (row1[i] == 1 and row2[i] == 1):
                loaded_feature = self._position_key[i]
            elif loaded_feature is not None and (row1[i] == 1 or row2[i] == 1) and not (row1[i] == 1 and row2[i] == 1):
                if loaded_feature == self._position_key[i]:
                    count += 1
        return count

    def _count_applicable_features(self, input_data, rule):
        n_applied = 0
        n_rules = 0
        for i in range(len(rule)):
            if rule[i] == 1:
                n_rules += 1
                #print('IAIN checking values: ', rule[i])
                #print(input_data[i])
                n_applied += 1 if input_data[i] == 1 else 0

        return n_applied / n_rules

    def _calculate_distance(self, rule, o_rule):
        distance = 0
        last_position = None
        for i in range(len(rule)):
            if rule[i] == 1:
                if last_position is not None:
                    distance += self._distance_matrix[i] - last_position
                    last_position = None
                else:
                    last_position = self._distance_matrix[i]
            if o_rule[i] == 1:
                if last_position is not None:
                    distance += self._distance_matrix[i] - last_position
                    last_position = None
                else:
                    last_position = self._distance_matrix[i]
        return distance

    def _get_similar_rules(self, data_row, rule, data_cls, new_class=0):
        # get rules concerned with the same feature but different values (statistically significant from contrast)
        in_vec, in_cls, pv, conf = rule
        #print("IAIN ENTERED SIMILAR CHECK")
        if in_cls == new_class:
            #print("DID NOT RETURN RULE ", in_cls, data_cls)
            return None
        similar_rule = None
        similar_pv = None
        similar_distance = None
        # first rule check, concerning exact rules (or contrasting rules)
        for i in range(len(self._all_rules)):
            rule_vec, rule_cls, rule_pv, rule_conf = self._all_rules[i]
            temp_applicability = self._count_applicable_features(data_row, rule_vec)
            #print("IAIN checking all rules: ", rule_cls, new_class)
            #print(np.any([(np.all(rule_vec == a) and rule_cls == b) for a, b, _, _ in self._applicable_rules]))
            if ((not np.any([(np.all(rule_vec == a) and rule_cls == b) for a, b, _, _ in self._applicable_rules])) and
                    rule_cls == new_class):
                if (temp_applicability != 1 and self._all_same_features(in_vec, rule_vec) and
                        (similar_rule is None or (similar_pv > rule_pv))):
                    #print("IAIN found rule")
                    temp_distance = self._calculate_distance(in_vec, rule_vec)
                    if similar_rule is None or similar_distance > temp_distance:
                        similar_rule = rule_vec
                        similar_pv = rule_pv
                        similar_distance = temp_distance

        #similar_rule = None
        #similar_pv = None
        #similar_distance = None
        if similar_rule is not None:
            #print('IAIN found good rule ', similar_rule)
            return similar_rule

        n_similar_features = 1
        # find somewhat similar rules to change with (may have strong impact)
        # rules should be similar and nearly applicable
        #print('IAIN checking other spots')
        best_feature_similarity = 0
        for i in range(len(self._all_rules)):
            rule_vec, rule_cls, rule_conf, rule_pv = self._all_rules[i]
            if ((not np.any([(np.all(rule_vec == a) and rule_cls == b) for a, b, _, _ in self._applicable_rules])) and
                    rule_cls == new_class):
                #print("IAIN found rule")
                temp_feature_similarity = self._count_same_features(in_vec, rule_vec)
                temp_applicability = self._count_applicable_features(data_row, rule_vec)
                #print('IAIN test counter', similar_pv, rule_pv)
                #print(temp_feature_similarity/np.nansum(rule[0]))
                #rint(temp_applicability)
                if (temp_applicability != 1 and temp_feature_similarity > best_feature_similarity and
                        ((temp_feature_similarity/np.nansum(rule[0]) > 0.2 and temp_applicability > 0.5)
                         or (1 > temp_applicability >= 0.5) and
                         (similar_rule is None or (similar_pv < rule_pv)))):
                    best_feature_similarity = temp_feature_similarity
                    similar_rule, _, _, similar_pv = self._all_rules[i]
                    n_similar_features = temp_feature_similarity

        # IAIN the third "do whatever" line of code is what added weird changes
        return similar_rule

    def _get_distance(self, row1, row2):
        # IAIN assumes that every rule compared has opening and closing values
        # use the positional differences to calculate a distance
        sum_dist = 0
        prev_feature = None
        for i in range(len(row1)):
            if (row1[i] == 1 or row2[i] == 1) and not (row1[i] == 1 and row2[i] == 1):
                if prev_feature is None:
                    prev_feature = i
                else:
                    # divide by the feature's standard deviation
                    sum_dist += abs(self._distance_matrix[i]-self._distance_matrix[prev_feature])
                    prev_feature = None

        return sum_dist

    def _same_feature(self, data_row, i):
        key_value = self._position_key[i]
        #print("IAIN KEY CHECK ", self._position_key)
        for j in range(len(data_row)):
            if data_row[j] == 1 and self._position_key[j] == key_value:
                #print("IAIN SAME FEATURE: ", i, "and", j)
                return True
        return False

    def predict(self, data_row, data_cls, new_class=0, n_counterfactuals=1):
        """
        Input: data_row (pandas DataFrame row) -> data row to find changes for.
               data_cls (int)                  -> class of row.
               new_class (int)                 -> desired new class.
                | Default: 0
        Purpose: Find applicable rules that need to be changed to try to change class.
        Output: new_data_row (pandas DataFrame row)    -> data row changed based on rules.
                rule_list (list<(rule_old, rule_new)>) -> rules that were changed and their replacements.
        """
        #print("IAIN in counterfactual predict")
        # IAIN take the value, find applicable rules and search for potential rule changes
        data_row = data_row[0]
        #print("IAIN APPLICABLE: ", self._applicable_rules)
        prev_applicable_rules = None
        new_data_values_list = []
        rule_changes_list = []
        # we want to find rules that are close to current rules that apply to the data_row but have the opposite
        #  prediction we will then move to those rows
        for k in range(n_counterfactuals):
            new_data_row = data_row.copy()
            n_loops = 0
            self._applicable_rules = self._get_applicable_rules(new_data_row, data_cls)
            rule_changes = []
            while ((prev_applicable_rules is None or
                    len(prev_applicable_rules) != len(self._applicable_rules) or
                    not np.all([prev_applicable_rules[i] == self._applicable_rules[i]
                               for i in range(len(self._applicable_rules))])) and n_loops < 4):
                # IAIN add doing one rule at a time
                original_rules = sorted(self._get_applicable_rules(new_data_row, data_cls), key=lambda x: abs(x[2]))
                for rule_translation in original_rules:
                    # TODO: fix issue with replacement using the already applicable rules
                    # IAIN should this call to the f(x) and get class at each step (or maybe to BARBE??)
                    # IAIN consider checking for the minimal rule that would change many and do that each loop instead...?

                    self._applicable_rules = sorted(self._get_applicable_rules(new_data_row, data_cls),
                                                    key=lambda x: abs(x[2]))
                    self._applicable_rules += sorted(self._get_applicable_rules(new_data_row, new_class),
                                                     key=lambda x: abs(x[2]))
                    prev_applicable_rules = self._applicable_rules
                    for rule_c in rule_changes_list:
                        for _, new_rule in rule_c:
                            # use this to ignore previous rules when generating multiples
                            self._applicable_rules.append((new_rule, new_class, None, None))
                    best_new_rule = self._get_similar_rules(data_row, rule_translation, data_cls, new_class=new_class)
                    self._applicable_rules.append((best_new_rule, new_class, None, None))

                    # change values that conflict with new rule
                    old_rule = rule_translation[0]
                    if best_new_rule is not None:
                        for i in range(len(new_data_row)):
                            #print(len(best_new_rule))
                            #print("IAIN FOUND RULES")
                            if (new_data_row[i] == 1 and self._same_feature(best_new_rule, i)
                                    and best_new_rule[i] != 1):
                                print("IAIN changed value at (to zero) ", i, ' for counter #', k)
                                new_data_row[i] = 0
                            elif new_data_row[i] == 0 and best_new_rule[i] == 1:
                                print("IAIN changed value at (to one) ", i, ' for counter #', k)
                                new_data_row[i] = 1
                    # track how rules were changed
                    rule_changes.append((rule_translation, best_new_rule))
                n_loops += 1
            new_data_values_list.append(new_data_row.copy())
            rule_changes_list.append(rule_changes.copy())
        # should return a change and what rules were modified
        #print(data_row)
        #print(n_loops)
        return ((new_data_values_list[0], rule_changes_list[0]) if n_counterfactuals == 1 else
                (new_data_values_list, rule_changes_list))


class BarbeCounterfactual:
    __doc__ = '''
        Purpose: Suggests modification of input data to change classification. Used and managed by BARBE explainer.

        Input: simple_distance (boolean) -> whether to use only simple distance comparison (no scale used).
                | Default: True
            '''

    def __init__(self, simple_distance=True):
        self._simple_distance = simple_distance

        self._distance_matrix = None
        # vector with importance values based on p-value * confidence
        self._importance_vector = None
        self._all_rules = None
        self._rule_classes = None
        self._feat_scales = None
        # key in the form [1, 2, 3, 1, 2, 1, 1, 1, ...]
        #                  binned         category
        self._position_key = None

    def _calculate_distance_matrix(self):
        self._distance_matrix = self._position_key
        j = 0
        for i in range(len(self._position_key)):
            if i + 1 < len(self._position_key) and self._position_key[i] <= self._position_key[i + 1]:
                self._position_key[i] = j
            else:
                self._position_key[i] = j
                j += 1

        self._position_key[-1] = j - 1

    def _calculate_importance_vector(self):
        if self._simple_distance:
            self._importance_vector = [1 for _ in range(len(self._position_key))]
        else:
            pass

    def _get_rule_pval(self, rule):
        return rule[4]

    def _get_rule_text(self, rule):
        return re.sub('->.*', '', rule[0])

    def _get_rule_class(self, rule):
        return rule[1]

    def _get_rule_len(self, rule):
        return len(rule[0].split(','))

    def _get_rule_check(self, rule, data_enc, check_type='applicable'):
        rule_enc = rule[0]
        if check_type == 'applicable':
            return np.sum(rule_enc & data_enc) == np.sum(rule_enc)
        elif check_type == 'number':
            return np.sum(rule_enc & data_enc)
        elif check_type == 'near-app':
            return np.sum(rule_enc) - np.sum(rule_enc & data_enc) <= 2 and np.sum(rule_enc) >= 1

        return np.any(rule_enc & data_enc)

    def _split_sort_classes(self, all_rules):
        # splits the rules based on class and then sorts the smaller lists
        class_dict = dict()
        for rule in all_rules:
            rule_c = self._get_rule_class(rule)
            if rule_c not in class_dict.keys():
                class_dict[rule_c] = list()
            class_dict[rule_c].append(rule)

        for c_val in class_dict.keys():
            class_dict[c_val].sort(key=lambda x: self._get_rule_pval(x))
        return class_dict

    def fit(self, ac_surrogate, feat_scales=1):
        """
        Input: rules (list<(enc_vec, cls, log(pval), 0)>) -> rules learned by BARBE.
               position_key (list<+int>)                      -> keys that show which feature each encoded value is.

        Purpose:

        Output:
        """
        self._ac_surrogate = ac_surrogate
        rules = self._ac_surrogate.get_all_rules(raw_rules=True)

        self._feature_scales = feat_scales
        self._rule_classes = self._split_sort_classes(rules)

        prev_feature = None
        feat_count = 0
        self._feature_map = list()
        for enc_name in self._ac_surrogate._oh_enc.get_feature_names_out():
            curr_feature = enc_name.split('_')[0]
            if curr_feature != prev_feature:
                feat_count += 1
            prev_feature = curr_feature
            self._feature_map.append(feat_count)


    def _get_applicable_rules(self, data_enc, cutoff_log_thresh=0):
        applicable_rules = dict()
        for rule_class in self._rule_classes.keys():
            applicable_rules[rule_class] = list()
            for rule in self._rule_classes[rule_class]:
                pval = self._get_rule_pval(rule)
                if pval < cutoff_log_thresh and self._get_rule_check(rule, data_enc):
                    applicable_rules[rule_class].append(rule)

        return applicable_rules

    def _calculate_distance(self, new_data, old_data):
        distance = 0
        return distance

    def _get_pval_totals(self, applicable_rules):
        pval_totals = defaultdict(int)
        print("TOTALS: ", applicable_rules)
        for rule_class in applicable_rules.keys():
            pval_totals[rule_class] = 0
            for rule in applicable_rules[rule_class]:
                pval_totals[rule_class] += self._get_rule_pval(rule)
        return pval_totals

    def _get_highest_class(self, pval_totals):
        highest_pval = 0
        highest_class = None
        for rule_class in pval_totals.keys():
            if highest_pval - pval_totals[rule_class] > 0:
                highest_class = rule_class
                highest_pval = pval_totals[rule_class]

        return highest_class

    def _get_feature_mask(self, single_rule_index):
        value_to_map = self._feature_map[single_rule_index]
        return np.array(self._feature_map) == value_to_map

    def _naive_replacement_applicable_impact(self, single_rule_index, applicable_rules, highest_class, new_class):
        impact_value = 0
        impact_positive = 0
        impact_negative = 0
        feature_masks = None
        if single_rule_index is not None:
            feature_masks = self._get_feature_mask(single_rule_index)

        for rule in applicable_rules[highest_class]:
            rule_item_check = True
            if single_rule_index is not None:
                rule_item_check = self._get_rule_check(rule, feature_masks, check_type='any')

            if rule_item_check:
                impact_value += self._get_rule_pval(rule)
                impact_positive += 1

        for rule in applicable_rules[new_class]:
            rule_item_check = True
            if single_rule_index is not None:
                rule_item_check = self._get_rule_check(rule, feature_masks, check_type='any')

            if rule_item_check:
                impact_value -= self._get_rule_pval(rule)
                impact_negative += 1

        return (impact_value,
                f'caused positive change in {impact_positive} rules and only hurt {impact_negative} rules')

    def _get_rule_distance(self, rule):
        # use the input distance values along with knowledge of categorical and not to
        #  adjust the prospective p-value (needs to be very good to warrant going far)
        pass

    def _naive_find_impacts(self, data_row, best_item, item_mask, check_class, class_sign,
                            impact_dict=None, change_dict=None):
        if impact_dict is None:
            impact_dict = dict()

        for rule in self._rule_classes[check_class]:
            # TODO: adjust near-app
            feature_matters = self._get_rule_check(rule, item_mask + (data_row & ~item_mask), check_type='near-app')
            if feature_matters:
                index_set = np.where(rule[0] & item_mask)[0]
                if len(index_set) > 0:
                    index_set = index_set[0]
                    if index_set not in impact_dict.keys():
                        impact_dict[index_set] = 0
                    impact_dict[index_set] += class_sign(self._get_rule_pval(rule))
        return impact_dict

    def _naive_replacement_all_impact(self, data_row, best_item, applicable_rules, highest_class, new_class):
        current_unlisted_features = list()
        rule_mask = self._get_feature_mask(best_item)
        possible_changes = dict()

        # get feature changes in list that would make data_row be put into highest_class
        feature_impacts = self._naive_find_impacts(data_row,
                                                   best_item,
                                                   rule_mask,
                                                   highest_class,
                                                   lambda x: -x)

        # find feature combinations that would put data_row into desired class
        # TODO: call to ideally the surrogate here (passed when fitting to extract encoder for changes
        #  (which BARBE will not handle))
        # TODO: the encoder should always handle this operation of checking input against rules -> or something else
        # TODO: the encoder should have the option to rough translate in order to know what the applied rule is during
        #  debugging
        feature_impacts = self._naive_find_impacts(data_row,
                                                   best_item,
                                                   rule_mask,
                                                   new_class,
                                                   lambda x: x,
                                                   impact_dict=feature_impacts)


        best_impact = feature_impacts[best_item]
        best_change = None
        # add nearest change later
        ordered_keys = list(feature_impacts.keys())
        ordered_keys.sort(key=lambda x: abs(x - best_item))
        for i in ordered_keys:
            if feature_impacts[i] < best_impact:
                best_impact = feature_impacts[i]
                best_change = i

        if best_change is not None:
            long_reason = (f'feature {best_change} has associated rules totalling pval {best_impact}'
                           f' where {best_item} has pval {feature_impacts[best_item]}')
        else:
            rule_mask[best_item] = 0
            best_change = np.where(rule_mask)[0][0]
            if best_change not in feature_impacts.keys() and best_impact > 0:
                best_impact = -best_impact
                long_reason = f'nothing was better than {best_item} so it was removed'
            else:
                long_reason = f'nothing was better than {best_item}'
                best_change = None

        return best_impact, best_change, long_reason

    def _get_rule_index(self, enc_rule):
        return np.where(enc_rule[0])

    def _find_naive_replacement_rule(self, data_row, applicable_rules, highest_class, new_class, rule_changes):
        replacement_rule = None
        replacement_reason = ''
        feature_to_change = ''
        rule_pos = 0
        best_item = None
        # get the current value (choose any that is better)
        best_impact, _ = self._naive_replacement_applicable_impact(None,
                                                                   applicable_rules,
                                                                   highest_class,
                                                                   new_class)
        original_impact = best_impact
        best_reason = ''
        impact_value = None
        best_long_impact = 0
        print("IN REPLACEMENT: ", len(applicable_rules[highest_class]))
        while rule_pos < len(applicable_rules[highest_class]):
            best_item = None
            rule_current = applicable_rules[highest_class][rule_pos]
            rule_index = self._get_rule_index(rule_current)[0]

            for single_rule_index in rule_index:
                # TODO: make this check the order of the change
                if np.any([rule_index != used_index[0] for used_index, _ in rule_changes]):
                    print([rule_index != used_index[0] for used_index, _ in rule_changes])
                    print(single_rule_index)
                    print(rule_changes)
                if np.all([rule_index != used_index[0] for used_index, _ in rule_changes]):
                    impact_value, impact_reason = self._naive_replacement_applicable_impact(single_rule_index,
                                                                                            applicable_rules,
                                                                                            highest_class,
                                                                                            new_class)
                    if best_impact - impact_value >= 0:
                        best_impact = impact_value
                        best_item = single_rule_index
                        best_reason = impact_reason

            if best_item is not None and best_impact <= original_impact:

                long_impact, long_change, long_reason = self._naive_replacement_all_impact(data_row,
                                                                                           best_item,
                                                                                           applicable_rules,
                                                                                           highest_class,
                                                                                           new_class)
                if long_change is not None and long_impact < best_long_impact:
                    best_long_impact = long_impact
                    replacement_rule = (best_item, long_change)
                    replacement_reason = long_reason
                print("IMPACT: ", best_impact, long_impact)
            rule_pos += 1

        return replacement_rule, replacement_reason

    def _find_replacement_rule(self, data_row, applicable_rules, highest_class, new_class, rule_changes,
                               selection_approach='naive'):
        # selection_approach = {naive, importance, graph}
        replacement_rule, replacement_reason = self._find_naive_replacement_rule(data_row, applicable_rules,
                                                                                 highest_class, new_class, rule_changes)
        # split to specific method based on selection approach
        #  naive select highest rule and the best rule that changes that value, selecting rules most
        #   similar to already applicable rules in the other class and changes the most other highest
        #   class rules.
        #  importance select value to change based on importance and choose a rule that makes the
        #   biggest difference and uses the smallest change in values.
        #  graph TBD.
        #
        # reason is compiled by each method based on what selection criteria was and what changed the
        #  candidate new rule when inspected (naive selected feature x, changed to y since x was in target
        #  class rules or was already changed previously ...)
        # compiling the final list of reasons into a clear and codified format that is human-readable
        #  so we can factor this into other research questions.
        return replacement_rule, replacement_reason

    def _apply_rule_change(self, enc_data, rule_change):
        rule_start, rule_end = rule_change
        enc_data[:, rule_start] = 0
        enc_data[:, rule_end] = 1
        return enc_data

    def _get_rule_changes(self, enc_data, in_class, new_class, applicable_rules, prev_applicable_rules,
                          rule_changes_list):
        new_data = enc_data.copy()
        rule_changes = list()
        pval_totals = self._get_pval_totals(applicable_rules)
        highest_class = in_class
        n_loops = 0
        print("TRYING TO GET RULE CHANGES = OLD: ", pval_totals[highest_class],
              " | NEW: ", pval_totals[new_class])
        rule_changes_flat = list()
        for list_change in rule_changes_list:
            rule_changes_flat += list_change
        replacement_rule = True
        while (n_loops < 5 and
               highest_class != new_class and
               pval_totals[highest_class] - pval_totals[new_class] < 0 and
               replacement_rule is not None):
            # passing the applicable rules means we can ignore the top rule and implement approaches here
            #  candidate_change_rule = applicable_rules[highest_class][0]
            # find the most similar rule in the new class (by features and distance)
            # naive approach considers the one rule alone, pass rule changes to not lose history
            replacement_rule, replacement_reason = self._find_replacement_rule(new_data,
                                                                               applicable_rules,
                                                                               highest_class,
                                                                               new_class,
                                                                               rule_changes_flat,
                                                                               selection_approach='naive')
            if replacement_rule is not None:
                rule_changes.append((replacement_rule, replacement_reason))
                rule_changes_flat.append((replacement_rule, replacement_reason))
                # change the data so the new rule applies
                new_data = self._apply_rule_change(new_data, replacement_rule)
                applicable_rules = self._get_applicable_rules(new_data)
                pval_totals = self._get_pval_totals(applicable_rules)
                highest_class = self._get_highest_class(pval_totals)

        return new_data, rule_changes

    def predict(self, in_data, new_class=0, n_counterfactuals=1):
        """
        Input: data_row (pandas DataFrame row) -> data row to find changes for.
               data_cls (int)                  -> class of row.
               new_class (int)                 -> desired new class.
                | Default: 0
        Purpose: Find applicable rules that need to be changed to try to change class.
        Output: new_data_row (pandas DataFrame row)    -> data row changed based on rules.
                rule_list (list<(rule_old, rule_new)>) -> rules that were changed and their replacements.
        """
        in_class = self._ac_surrogate.predict(in_data.to_numpy().reshape((1, -1)))[0]
        enc_data = self._ac_surrogate._encode(in_data.to_numpy().reshape((1, -1)).copy())

        prev_applicable_rules = list()
        new_data_values_list = []
        rule_changes_list = []
        # we want to find rules that are close to current rules that apply to the data_row but have the opposite
        #  prediction we will then move to those rows
        print("CALL APPLICABLE")
        applicable_rules = self._get_applicable_rules(enc_data)

        prev_applicable_rules.append(applicable_rules)
        for k in range(n_counterfactuals):
            new_data, rule_changes = self._get_rule_changes(enc_data,
                                                            in_class, new_class,
                                                            applicable_rules, prev_applicable_rules,
                                                            rule_changes_list)

            new_applicable_rules = self._get_applicable_rules(new_data)
            new_data_values_list.append(self._ac_surrogate._decode(new_data))
            rule_changes_list.append(rule_changes)
            prev_applicable_rules.append(new_applicable_rules)


        return ((new_data_values_list[0], rule_changes_list[0]) if n_counterfactuals == 1 else
                (new_data_values_list, rule_changes_list))

