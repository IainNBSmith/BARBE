################################################################################
# Copyright (C) 2020 Mohammad Motallebi
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

from libcpp cimport bool
from libcpp cimport int
from libcpp cimport float
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

from cython.operator import dereference, postincrement

import numpy as np
from collections import Counter, defaultdict

# Python class Rule
import sigdirect.rule

# enum heuristics
cdef extern from "include/sigdirect.h":
    cdef enum Heuristic:
        hrs1,
        hrs2,
        hrs3,

# CPP class Rule 
cdef extern from "src/rule.h":
    cppclass Rule:
        Rule();
        vector[int] get_items() const; # internal  SD features
        vector[int] get_original_items() const; # user features
        int get_label() const; # SD internal label
        int get_original_label() const; # user label
        double get_support() const;
        long double get_confidence() const;
        long double get_p() const;
        long double get_log_p() const;
        bool applies_to(unordered_set[int]);

        long double get_confidence_prune() const;

        int get_importance() const;

# CPP Sigdirect
cdef extern from "include/sigdirect.h":
    cppclass CPPSigDirect:
        CPPSigDirect();
        CPPSigDirect(int, double, bool, double);
        
        # train
        pair[int,int] fit( vector[vector[int]], vector[int])
        
        # test
        vector[int] predict(const vector[vector[int]]&, const Heuristic&);
        vector[long double] predict_proba();
        unordered_map[int, vector[Rule]] get_all_rules();

# Python SigDirect
cdef class SigDirect:

    cdef _all_rules
    cdef _is_transaction_form
    cdef _clf_version
    cdef _alpha
    cdef _early_stopping
    cdef _confidence_threshold
    cdef _hrs
    cdef _majority_class
    cdef _other_info

    def __init__(self, 
                clf_version=1, 
                alpha=0.05, 
                early_stopping=False, 
                confidence_threshold=0.5, 
                is_binary=False, 
                get_logs=False,
                other_info=None):
        self._clf_version = clf_version
        self._alpha = alpha
        self._early_stopping = early_stopping
        self._confidence_threshold = confidence_threshold
        self._is_transaction_form = not is_binary
        self._all_rules = defaultdict(list)
        self._hrs = 0 # will be set in predict()
        self._other_info = other_info

    def fit(self, X, y):
        cobj = new CPPSigDirect(self._clf_version,
                                self._alpha, 
                                self._early_stopping, 
                                self._confidence_threshold)
        cdef vector[vector[int]] input_X
        if not self._is_transaction_form:
            for i in range(X.shape[0]):
                input_X.push_back(np.nonzero(X[i])[0])
        else:
            input_X = X

        # train the classifier
        self._majority_class = max(list(Counter(y).items()), key=lambda x:x[1])[0]
        ret = cobj.fit(input_X,y)

        # store rules in python variables
        rules_cpp = cobj.get_all_rules()
        cdef unordered_map[int, vector[Rule]].iterator it = rules_cpp.begin()
        while it!=rules_cpp.end():
            p = []
            for i in range(dereference(it).second.size()):
                if self._other_info:
                    new_rule = rule.Rule(dereference(it).second[i].get_original_items(), 
                                    dereference(it).second[i].get_original_label(),
                                    dereference(it).second[i].get_confidence(),
                                    dereference(it).second[i].get_log_p(),
                                    dereference(it).second[i].get_support(),
                                    dereference(it).second[i].get_importance(),
                                    self._other_info['local_id2words'],
                                    )
                else:
                   new_rule = rule.Rule(dereference(it).second[i].get_original_items(),
                                    dereference(it).second[i].get_original_label(),
                                    dereference(it).second[i].get_confidence(),
                                    dereference(it).second[i].get_log_p(),
                                    dereference(it).second[i].get_support(),
                                    dereference(it).second[i].get_importance(),
                                    )

                p.append(new_rule)
            self._all_rules[dereference(it).first] = p
            postincrement(it)

        # get rid of cpp object
        del cobj

        return ret

    def get_all_rules(self):
        return self._all_rules

    def predict(self, X, hrs=1):
        """ Given a list of instances, predicts their corresponding class
        labels and returns the labels.

        Args:
        X: test instances in the form of a 2-d numpy array
        heuristic: the heuristic used in classification (1, 2, 3)

        Returns:
        a list of labels corresponding to all instances.
        """
        self._hrs = hrs

        if hrs not in (1,2,3):
            raise Exception("heuristic value should either be 1, 2, or 3")

        if type(X) == np.ndarray:
            if len(X.shape)>2:
                raise Exception("2-d numpy array expected")
            if len(X.shape)==2:
                predictions = np.apply_along_axis(self._predict_instance, axis=1, arr=X)
            else:
                predictions = self._predict_instance(X)
        elif type(X) == list:
            predictions = self._predict_instance(X)
        else:
            raise TypeError("Invalid data type detected in predict function")

        return np.array(predictions)


    def _predict_instance(self, instance):
        hrs = self._hrs
        if not self._is_transaction_form:
            instance = np.nonzero(instance)[0]

        # for each label compute the corresponding score.
        all_labels = self.get_all_rules().keys()
        scores = [(self._get_similarity_to_label(instance, x), x) for x in all_labels]
        
        # no applicable rule
        if sum([x[0] for x in scores])==0.0:
            return self._majority_class

        # find best score based on heuristic
        return self._get_best_match_label(scores)

    def _get_similarity_to_label(self, instance, label):
        hrs = self._hrs

        heuristic_funcs = [SigDirect._hrs_1, SigDirect._hrs_2, SigDirect._hrs_3]

        sum_ = 0.0
        for rule in self.get_all_rules()[label]:
            if SigDirect._rule_matches(instance, rule):
                sum_ += heuristic_funcs[hrs-1](rule) * float(rule.get_importance())

        return sum_

    def _get_best_match_label(self, scores):
        hrs = self._hrs

        min_ = min(scores, key=lambda x:(x[0],x[1]!=self._majority_class))
        max_ = max(scores, key=lambda x:(x[0],x[1]==self._majority_class))

        if hrs in [1,3]:# these heuristics look for minimum score
            return min_[1]
        else:
            return max_[1]
    
    @staticmethod
    def _rule_matches(instance, rule):
        instance_items_set = set(instance)
        for id_item in rule.get_items():
            if id_item not in instance_items_set:
                return False
        return True

    @staticmethod
    def _hrs_1(rule):
        return rule.get_log_p()
        # x = rule.get_ss()
        # if x==0.0:
        #     return -float('inf')            
        # if x> 2*-500:
        #     return np.log(x)
        # else:
        #     return -float('inf')

    @staticmethod
    def _hrs_2(rule):
        return rule.get_confidence()

    @staticmethod
    def _hrs_3(rule):
        return rule.get_log_p() * rule.get_confidence()
        # x = rule.get_ss()
        # if x==0.0:
        #     return -float('inf')            
        # if x > 2*-500:
        #     return float(np.log(x)) * rule.get_confidence()
        # else:
        #     return -float('inf')
