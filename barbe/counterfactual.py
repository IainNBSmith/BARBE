import numpy as np


# TODO: add distance information to counterfactual

class BarbeCounterfactual:
    __doc__ = """
    
    """

    def __init__(self, simple_distance=True):
        self._simple_distance = simple_distance

        self._distance_matrix = None
        # vector with importance values based on p-value * confidence
        self._importance_vector = None
        self._all_rules = None
        # key in the form [1, 2, 3, 1, 2, 1, 1, 1, ...]
        #                  binned         category
        self._position_key = None
        pass

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
        #print("IAIN FINAL POSITION KEY: ", self._position_key)

    def _calculate_importance_vector(self):
        if self._simple_distance:
            self._importance_vector = [1 for _ in range(len(self._position_key))]
        else:
            pass

    def fit(self, rules, position_key):
        # rules = list<array(features), label, p_val, confidence>
        # IAIN take all the rules that matter and use them for reasoning
        # create distance calculating matrices (p_val * confidence)
        # store potential modifications (matrix)
        print("IAIN in fitting counterfactual")
        self._all_rules = rules
        self._position_key = position_key

        self._calculate_distance_matrix()
        self._calculate_importance_vector()

    def _get_applicable_rules(self, data_row, data_cls):
        # IAIN get all applicable rules for the given row (what to change)
        # rule<(rule vec, class, conf, supp, pval)>
        # for rule_vec, rule_cls, rule_cnf, rule_sup, rule_pv in self._all_rules:
        applicable_rules = []
        for i in range(len(self._all_rules)):
            print(self._all_rules[i])
            rule_vec, rule_cls, _, _ = self._all_rules[i]
            if rule_cls == data_cls:
                print("IAIN FOUND POTENTIAL RULE")
                print([(rule_vec[j] == data_row[j] or (rule_vec[j] == 0 and self._distance_matrix[j] != 1)) for j in range(len(data_row))])
                if np.array([(rule_vec[j] == data_row[j] or (rule_vec[j] == 0)) for j in range(len(data_row))]).all():
                    applicable_rules.append(self._all_rules[i])

        return applicable_rules

    def _all_same_features(self, row1, row2):
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

    def _get_similar_rules(self, rule, data_cls):
        # get rules concerned with the same feature but different values (statistically significant from contrast)
        in_vec, in_cls, conf, _ = rule
        print("IAIN ENTERED SIMILAR CHECK")
        if in_cls != data_cls:
            print("DID NOT RETURN RULE ", in_cls, data_cls)
            return None
        similar_rule = None
        similar_pv = None
        for i in range(len(self._all_rules)):
            rule_vec, rule_cls, rule_conf, rule_pv = self._all_rules[i]
            if rule_cls != in_cls:
                if self._all_same_features(in_vec, rule_vec) and (similar_rule is None or (similar_pv >
                                                                                           rule_pv/rule_conf)):
                    similar_rule = rule_vec
                    similar_pv = rule_pv/rule_conf

        if similar_rule is not None:
            return similar_rule

        n_similar_features = 1
        for i in range(len(self._all_rules)):
            rule_vec, rule_cls, rule_conf, rule_pv = self._all_rules[i]
            if rule_cls != in_cls:
                temp_feature_similarity = self._count_same_features(in_vec, rule_vec)
                if temp_feature_similarity > n_similar_features and (similar_rule is None or (similar_pv > rule_pv)):
                    similar_rule, _, _, similar_pv = self._all_rules[i]
                    n_similar_features = temp_feature_similarity

        if similar_rule is not None:
            return similar_rule

        for i in range(len(self._all_rules)):
            rule_vec, rule_cls, rule_conf, rule_pv = self._all_rules[i]
            if rule_cls != in_cls:
                temp_feature_similarity = self._count_same_features(in_vec, rule_vec)
                if similar_rule is None or (similar_pv > rule_pv):
                    similar_rule, _, _, similar_pv = self._all_rules[i]
                    n_similar_features = temp_feature_similarity

        print("IAIN DID NOT FIND RULES")
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
        print("IAIN KEY CHECK ", self._position_key)
        for j in range(len(data_row)):
            if data_row[j] == 1 and self._position_key[j] == key_value:
                print("IAIN SAME FEATURE: ", i, "and", j)
                return True
        return False

    def predict(self, data_row, data_cls):
        print("IAIN in counterfactual predict")
        # IAIN take the value, find applicable rules and search for potential rule changes
        data_row = data_row[0]
        applicable_rules = self._get_applicable_rules(data_row, data_cls)
        print("IAIN APPLICABLE: ", applicable_rules)
        prev_applicable_rules = None
        n_loops = 0
        new_data_row = data_row.copy()
        # we want to find rules that are close to current rules that apply to the data_row but have the opposite
        #  prediction we will then move to those rows
        rule_changes = []

        while prev_applicable_rules != applicable_rules and n_loops < 1:
            for rule_translation in applicable_rules:
                print("IAIN finding new rule")
                swapped_rules = self._get_similar_rules(rule_translation, data_cls)
                print("IAIN SWAPPED: ", swapped_rules)
                best_rule_distance = None
                best_new_rule = None

                # choose the best new rule
                '''
                for new_rule, _ in swapped_rules:
                    temp_distance = self._get_distance(rule_translation, new_rule)
                    if best_rule_distance is None or temp_distance < best_rule_distance:
                        best_rule_distance = temp_distance
                        best_new_rule = new_rule
                '''
                best_new_rule = swapped_rules

                # change values that conflict with new rule
                old_rule = rule_translation[0]
                if best_new_rule is not None:
                    for i in range(len(new_data_row)):
                        print(len(best_new_rule))
                        print("IAIN FOUND RULES")
                        if (new_data_row[i] == 1 and self._same_feature(old_rule, i)
                                and best_new_rule[i] != 1):
                            print("IAIN changed value at (to zero) ", i)
                            new_data_row[i] = 0
                        elif new_data_row[i] != 1 and best_new_rule[i] == 1:
                            print("IAIN changed value at (to one) ", i)
                            new_data_row[i] = 1
                # track how rules were changed
                rule_changes.append((rule_translation, best_new_rule))
            n_loops += 1
            prev_applicable_rules = applicable_rules
            applicable_rules = self._get_applicable_rules(new_data_row, data_cls)

        # should return a change and what rules were modified
        print(data_row)
        print(n_loops)
        return new_data_row, rule_changes
