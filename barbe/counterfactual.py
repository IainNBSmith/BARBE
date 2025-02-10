import numpy as np


# TODO: add distance information to counterfactual
# TODO: check why even if you pass the current class it will still suggest a change (maybe I am too aggressive with changes?)
# TODO: Add input for rule changes that are allowed dict{feat_1:0, ...} 0 -> no changes, 1 -> + only, 2 -> - only, 3 -> +/-
#  TODO: we do this cause then if dict[feat1]: make change (then more checking for specifics)
# TODO: include option setting for how intense rule changes should be (e.g. only contrast or further...)
# TODO: Print the applicable rules and all rules into another file that we may review
# TODO: For the counterfactuals print the applicable rules into another file to to see why the prediction was made in one way
# TODO: Clean up test code so everything is clear
# TODO: Start implementing the test/trial for number classification comparison with DiCE

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
        Input: rules (list<(enc_vec, cls, supp, conf, pval)>) -> rules learned by BARBE.
               position_key (list<+int>)                      -> keys that show which feature each encoded value is.
        Purpose:
        Output:
        """
        self._all_rules = rules
        self._position_key = position_key

        self._calculate_distance_matrix()
        self._calculate_importance_vector()

    def _get_applicable_rules(self, data_row, data_cls):
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

                if np.array([(rule_vec[j] == data_row[j] or (rule_vec[j] == 0)) for j in range(len(data_row))]).all():
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
        in_vec, in_cls, conf, pv = rule
        #print("IAIN ENTERED SIMILAR CHECK")
        if in_cls == new_class:
            #print("DID NOT RETURN RULE ", in_cls, data_cls)
            return None
        similar_rule = None
        similar_pv = None
        similar_distance = None
        # first rule check, concerning exact rules (or contrasting rules)
        for i in range(len(self._all_rules)):
            rule_vec, rule_cls, rule_conf, rule_pv = self._all_rules[i]
            temp_applicability = self._count_applicable_features(data_row, rule_vec)
            #print("IAIN checking all rules: ", rule_cls, new_class)
            #print(np.any([(np.all(rule_vec == a) and rule_cls == b) for a, b, _, _ in self._applicable_rules]))
            if ((not np.any([(np.all(rule_vec == a) and rule_cls == b) for a, b, _, _ in self._applicable_rules])) and
                    rule_cls == new_class):
                if (temp_applicability != 1 and self._all_same_features(in_vec, rule_vec) and
                        (similar_rule is None or (similar_pv > rule_pv/(rule_conf+1e-16)))):
                    #print("IAIN found rule")
                    temp_distance = self._calculate_distance(in_vec, rule_vec)
                    if similar_rule is None or similar_distance > temp_distance:
                        similar_rule = rule_vec
                        similar_pv = rule_pv/(rule_conf + 1e-16)
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
            if row1[i] == 1 or row2[i] == 1 and not (row1[i] == 1 and row2[i] == 1):
                if prev_feature is None:
                    prev_feature = i
                else:
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
            while prev_applicable_rules != self._applicable_rules and n_loops < 2:
                # IAIN add doing one rule at a time
                original_rules = self._get_applicable_rules(new_data_row, data_cls)
                for rule_translation in original_rules:
                    # TODO: fix issue with replacement using the already applicable rules
                    # IAIN should this call to the f(x) and get class at each step (or maybe to BARBE??)
                    # IAIN consider checking for the minimal rule that would change many and do that each loop instead...?

                    self._applicable_rules = self._get_applicable_rules(new_data_row, data_cls)
                    self._applicable_rules += self._get_applicable_rules(new_data_row, new_class)
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
                                #print("IAIN changed value at (to zero) ", i)
                                new_data_row[i] = 0
                            elif new_data_row[i] != 1 and best_new_rule[i] == 1:
                                #print("IAIN changed value at (to one) ", i)
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
