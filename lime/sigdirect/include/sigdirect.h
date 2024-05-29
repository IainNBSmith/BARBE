/*******************************************************************************
 * Copyright (C) 2020 Mohammad Motallebi
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#ifndef _SIGDIRECT
#define _SIGDIRECT

#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config.h"
#include "node.h"
#include "rule.h"
#include "rule_node.h"
using namespace std;

enum Heuristic { hrs1 = 1, hrs2, hrs3 };

class CPPSigDirect {
 private:
  int majority_class;
  static uint16_t debug_counter;
  unordered_map<int, vector<Rule*>> initial_rules;  // before prune
  unordered_map<int, vector<Rule*>> final_rules;    // after prune
  int features_size;
  Heuristic hrs;
  vector<vector<int>> transactions;
  vector<int> transaction_classes;
  vector<vector<int>> transactions_prune;  // SigD2
  vector<int> transaction_prune_classes;   // SigD2
  set<uint16_t> train_indexes;             // SigD2

  Node* root;
  unordered_map<int, int> labels_support;
  unordered_map<int, double> labels_support_lgamma;
  int d_size;
  double d_size_lgamma;

  // original label (given by user) --> SigDirect's (0-index)
  unordered_map<int, int> class_map;
  unordered_map<int, int> class_rmap;  // reverse

  // original instance feature --> SigDirect's (0-index & ascending order)
  unordered_map<int, int> feature_map;
  unordered_map<int, int> feature_rmap;  // reverse

  void release_memory();  // needed because of Cython

  /* Dataset preparation */
  int get_label_support(const int&);
  double get_label_support_lgamma(const int&);
  void set_label_support(vector<int>);
  void preprocess_data(const vector<vector<int>>&);
  void preprocess_labels(const vector<int>&);
  vector<vector<int>> transform_data(const vector<vector<int>>&);
  vector<int> transform_labels(const vector<int>&);
  vector<vector<int>> untransform_data(const vector<vector<int>>&);
  vector<int> untransform_labels(const vector<int>&);
  void set_majority_class(const vector<int>&);

  /* Tree generation */
  void build_tree();
  void deepen_tree(const int&, int&, int&);
  void step_traverse(const vector<int>&, const int&, Node*, uint16_t,
                     const uint16_t&, uint16_t, int&, int&);
  void set_pvalues(int);
  void is_rule_pss_and_not_minimal(const vector<uint16_t>&, const int&,
                                   const uint16_t&, pair<bool, double>&);
  void are_parents_pss_and_not_minimal_and_min_pss(const vector<uint16_t>&,
                                                   const int&,
                                                   pair<bool, double>&);
  void set_rule_node_pvalue(RuleNode*, const int&, const int&);
  void set_pvalues_node_1(const int&, Node*, const vector<uint16_t>&);
  void set_pvalues_node_2(const int&, Node*, const vector<uint16_t>&);
  void set_pvalues(const int&, int, Node*, vector<uint16_t>&);

  /* Prune rules */
  void prune_rules_sd(vector<vector<int>>, vector<int>,
                      unordered_map<int, vector<Rule*>>);
  void prune_rules_sigd2(vector<vector<int>>, vector<int>,
                         unordered_map<int, vector<Rule*>>);
  void update_confidence_prune(vector<unordered_set<int>>, vector<int>,
                               const vector<bool>&, vector<Rule*>);

  /* Prediction */
  int predict_instance(unordered_set<int>, Heuristic);
  vector<Rule*> get_applicable_rules(unordered_set<int>, vector<Rule*>);

 public:
  CPPSigDirect(int, double, bool, double);
  CPPSigDirect();
  ~CPPSigDirect();

  // train
  pair<int, int> fit(vector<vector<int>>, vector<int> v);

  // test
  vector<int> predict(const vector<vector<int>>&,
                      const Heuristic& hrs = Heuristic::hrs2);
  vector<double> predict_proba() { throw "not implemented error"; }

  unordered_map<int, vector<Rule>> get_all_rules();
};

#endif
