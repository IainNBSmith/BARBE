////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2020 Mohammad Motallebi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////////////
/*
 NOTE:
 1. class label are currently changed to 0-based indexes only in class_map
 2. the count of each rule_node is limited to uint16_t
 3. the count of each node is limited to uint16_t
 4. the items numbers are limited to uint16_t (~65k)

 - input data is read from txt file and written to data (2d vector),
 - it is then converted to sparse (one-hot enocoded) format, and sent to fit
 - inside fit. it is converted back to vectors. [?]

 */

#include "sigdirect.h"
using namespace std;

struct numbers config;
uint16_t CPPSigDirect::debug_counter = 0;

CPPSigDirect::CPPSigDirect(int clf_version, double alpha = 0.05,
                           bool early_stopping = false,
                           double confidence_threshold = 0.5) {
  static bool once = []() {
    init(config.log_severity, config.log_filename.c_str(), 1000000, 1);
    return true;
  }();
  if (false && once)
    ;  // get rid of warning
  PLOG_INFO << "NEW RUN: " << ++CPPSigDirect::debug_counter;

  config.clf_version = clf_version;
  config.alpha = alpha;
  config.log_alpha = log(alpha);
  config.early_stopping = early_stopping;
  config.confidence_threshold = confidence_threshold;
}

CPPSigDirect::CPPSigDirect() {
  static bool once = []() {
    init(config.log_severity, config.log_filename.c_str(), 1000000, 1);
    return true;
  }();
  if (false && once)
    ;  // get rid of warning
  PLOG_INFO << "NEW RUN: " << ++CPPSigDirect::debug_counter;
}

CPPSigDirect::~CPPSigDirect() {
  if (this->root != NULL) {
    delete this->root;
    this->root = NULL;
  }
}

void CPPSigDirect::release_memory() {
  if (this->root != NULL) {
    delete this->root;
    this->root = NULL;
  }
}

/* Dataset preparation */
vector<vector<int>> CPPSigDirect::transform_data(const vector<vector<int>> &X) {
  /* for each input data, convert its features to SD features. */
  vector<vector<int>> ret;
  for (auto instance : X) {
    vector<int> temp;
    for (auto f : instance) {
      temp.push_back(this->feature_map[f]);
    }
    sort(temp.begin(), temp.end());
    ret.push_back(temp);
  }
  return ret;
}

vector<int> CPPSigDirect::transform_labels(const vector<int> &y) {
  /* for each input label, convert it to SD label. */
  vector<int> ret(y.size());
  for (size_t i = 0; i < y.size(); i++) ret[i] = this->class_map[y[i]];
  return ret;
}

void CPPSigDirect::set_majority_class(const vector<int> &y) {
  unordered_map<int, int> freqs_map;
  for (int label : y) {
    if (freqs_map.find(label) == freqs_map.end())
      freqs_map[label] = 1;
    else
      freqs_map[label]++;
  }
  auto p =
      max_element(freqs_map.begin(), freqs_map.end(),
                  [](auto lhs, auto rhs) { return lhs.second < rhs.second; });
  this->majority_class =
      this->class_map[p->first];  // convert label to SD label
}

vector<vector<int>> CPPSigDirect::untransform_data(
    const vector<vector<int>> &X) {
  vector<vector<int>> ret;
  for (auto instance : X) {
    vector<int> temp;
    for (auto f : instance) {
      temp.push_back(this->feature_rmap[f]);
    }
    ret.push_back(temp);
  }
  return ret;
}

vector<int> CPPSigDirect::untransform_labels(const vector<int> &y) {
  vector<int> ret(y.size());
  for (size_t i = 0; i < y.size(); i++) ret[i] = this->class_rmap[y[i]];
  return ret;
}

void CPPSigDirect::preprocess_data(const vector<vector<int>> &X) {
  // save how many occurrence of each feature we have.
  unordered_map<int, int> feature_count;
  for (auto instance : X) {
    for (auto f : instance) {
      if (feature_count.find(f) == feature_count.end())
        feature_count[f] = 1;
      else
        feature_count[f]++;
    }
  }

  // re-arrange features in an increasing order
  vector<pair<int, int>> temp(feature_count.begin(), feature_count.end());
  sort(temp.begin(), temp.end(),
       [](auto a, auto b) { return a.second < b.second; });
  for (size_t i = 0; i < temp.size(); i++) {
    this->feature_map[temp[i].first] = i;
    this->feature_rmap[i] = temp[i].first;
  }

  if (config.clf_version == 2) {
    //		while(this->train_indexes.size()<2*X.size()/3){
    //			uint16_t new_index = (uint16_t) rand() % X.size();
    ////			cout << new_index  << endl;
    ////			this->train_indexes.insert(new_index);
    //		}
    vector<vector<int>> temp1, temp2;
    for (uint16_t index = 0; index < X.size(); index++) {
      if (index % 3 != 0) {
        //			if(this->train_indexes.find(index)!=this->train_indexes.end())
        temp1.push_back(X[index]);
        this->train_indexes.insert(index);
      } else
        temp2.push_back(X[index]);
    }
    this->transactions = transform_data(temp1);
    this->transactions_prune = transform_data(temp2);
  } else {
    // for each input data, convert its features to SD features.
    this->transactions = transform_data(X);
  }
}

void CPPSigDirect::preprocess_labels(const vector<int> &y) {
  int label_counter = 0;
  for (auto label : y) {
    if (this->class_map.find(label) == this->class_map.end()) {
      this->class_map[label] = label_counter;
      label_counter++;
    }
  }

  for (auto p : this->class_map) {
    if (this->class_rmap.find(p.second) != this->class_rmap.end())
      throw "something is wrong with class labels!";
    this->class_rmap[p.second] = p.first;
  }

  if (config.clf_version == 2) {  // SigD2
    vector<int> temp1, temp2;
    for (uint16_t index = 0; index < y.size(); index++) {
      if (this->train_indexes.find(index) != this->train_indexes.end())
        temp1.push_back(y[index]);
      else
        temp2.push_back(y[index]);
    }
    this->transaction_classes = transform_labels(temp1);
    this->transaction_prune_classes = transform_labels(temp2);
  } else {
    this->transaction_classes = transform_labels(y);
  }

  set_majority_class(y);
}

int CPPSigDirect::get_label_support(const int &label) {
  unordered_map<int, int>::iterator p = this->labels_support.find(label);
  if (p == this->labels_support.end()) throw "label not found!!";
  return p->second;
}

double CPPSigDirect::get_label_support_lgamma(const int &label) {
  unordered_map<int, double>::iterator p =
      this->labels_support_lgamma.find(label);
  return p->second;
}

void CPPSigDirect::set_label_support(vector<int> mapped_y) {
  for (int label : mapped_y) {
    auto p = this->labels_support.find(label);
    if (p == this->labels_support.end())
      this->labels_support[label] = 1;
    else
      this->labels_support[label]++;
  }
  for (auto p : this->labels_support) {
    PLOG_WARNING << "LABEL SIZE " << p.first << ": " << p.second;
    this->labels_support_lgamma[p.first] = lgamma(p.second + 1);
  }
}

/* Tree generation */
pair<int, int> CPPSigDirect::fit(vector<vector<int>> X, vector<int> y) {
  /* args:
   * 	X: 2d vector of int (present features in each instance)
   * 	y: 1d vector of int (class labels from user)
   * return:
   * 	pair<int,int>: # initial rules, # final rules (debug)
   */

  // process X to get this->transactions, and corresponding maps
  preprocess_data(X);

  // process y to get this->transaction_classes, and corresponding maps
  preprocess_labels(y);

  // set the labels_support map
  set_label_support(this->transaction_classes);
  assert(this->transactions.size() == this->transaction_classes.size());

  this->d_size = this->transactions.size();
  this->d_size_lgamma = lgamma(this->d_size + 1);
  Node::labels_size = this->class_map.size();

  this->root = new Node();
  build_tree();

  // prune useless rules
  if (config.clf_version == 2) {  // SigD2, internally calls prune_rules_sd()-
    prune_rules_sigd2(this->transactions_prune, this->transaction_prune_classes,
                      this->initial_rules);
  } else {  // original SigDirect
    prune_rules_sd(this->transactions, this->transaction_classes,
                   this->initial_rules);
  }

  // print rules (debugging)
  ofstream fout;
  fout.open("rules_out.txt", std::ios_base::app);
  for (auto p : this->final_rules) {
    for (auto r : p.second) {
      //      r->print_raw_rule();
      r->print_rule(this->feature_rmap, class_rmap, fout);
    }
  }
  fout << endl;
  fout.close();

  this->release_memory();

  // return number of initial/final_rules (debugging)
  return pair<int, int>(
      accumulate(this->initial_rules.begin(), this->initial_rules.end(), 0,
                 [](const size_t previous, const pair<int, vector<Rule *>> &p) {
                   return previous + p.second.size();
                 }),
      accumulate(this->final_rules.begin(), this->final_rules.end(), 0,
                 [](const size_t previous, const pair<int, vector<Rule *>> &p) {
                   return previous + p.second.size();
                 }));
}

void CPPSigDirect::build_tree() {
  int max_depth = 0;
  int rule_nodes_count = 0, nodes_count = 0;
  int last_rules_count = -1;

  while (max_depth == 0 || rule_nodes_count > 0) {
    nodes_count = 0;
    rule_nodes_count = 0;

    max_depth++;
    vector<uint16_t> deepen_t(max_depth - 1, 0);
    vector<uint16_t> t(max_depth, 0);  // placeholder for curr_items
    deepen_tree(max_depth, nodes_count, rule_nodes_count);

    set_pvalues(max_depth, 0, this->root, t);

    int rules_size = accumulate(
        this->initial_rules.begin(), this->initial_rules.end(), 0,
        [](int previous, auto p) { return previous + p.second.size(); });

    PLOG_INFO << max_depth << " ---new nodes: " << nodes_count
              << " new rule_nodes: " << rule_nodes_count
              << " rules: " << rules_size;

    // early stopping
    if (config.early_stopping && rules_size == last_rules_count) break;
    last_rules_count = rules_size;

    if (max_depth > config.max_depth) break;
  }
}

void CPPSigDirect::deepen_tree(const int &max_depth, int &nodes_count,
                               int &rule_nodes_count) {
  /* expand the tree by one layer */
  Node *current_node = this->root;
  int current_depth = 1;
  int index = 0;
  for (size_t i = 0; i < this->transactions.size(); i++) {
    step_traverse(transactions[i], this->transaction_classes[i], current_node,
                  current_depth, max_depth, index, rule_nodes_count,
                  nodes_count);
  }
}

void CPPSigDirect::step_traverse(const vector<int> &transaction,
                                 const int &label, Node *current_node,
                                 uint16_t current_depth,
                                 const uint16_t &max_depth, uint16_t index,
                                 int &rule_nodes_count, int &nodes_count) {
  /* process one instance to expand the tree*/

  // have reached max_depth or will not reach it anymore
  if (current_depth > max_depth) {
    return;
  }
  if (index == transaction.size()) {
    return;
  }
  if (transaction.size() + current_depth < max_depth + index) {  // TODO
    return;
  }

  if (current_depth < max_depth) {  // traverse the tree
    if (current_depth == 1 ||
        (current_node->has_label(label) &&
         !current_node->get_rule_node(label)->get_is_minimal()))
      for (uint16_t i = index;
           i < transaction.size() + current_depth - max_depth; i++) {
        if (current_node->has_child(transaction[i])) {
          step_traverse(transaction, label,
                        current_node->get_child(transaction[i]),
                        current_depth + 1, max_depth, i + 1, rule_nodes_count,
                        nodes_count);
        }
      }
  } else {  // last layer
    Node *next_node;
    RuleNode *next_rule_node;
    for (size_t i = index; i < transaction.size(); i++) {
      if (!current_node->has_child(transaction[i])) {
        next_node = new Node();
        nodes_count++;
        current_node->add_child(transaction[i], next_node);
      } else {
        next_node = current_node->get_child(transaction[i]);
      }

      if (!next_node->has_label(label)) {
        next_rule_node = next_node->add_rule_node(label);
        rule_nodes_count++;
      } else {
        next_rule_node = next_node->get_rule_node(label);
      }
      next_node->increase_count();
      next_rule_node->increase_count();
    }
  }
}

void CPPSigDirect::set_pvalues_node_1(const int &curr_depth, Node *curr_node,
                                      const vector<uint16_t> &curr_items) {
  auto p_temp = curr_node->get_rule_nodes();
  for (size_t i = 0; i < this->class_map.size(); i++) {
    if (p_temp[i] == NULL) continue;
    int label = i;
    auto rule_node = p_temp[i];

    rule_node->set_is_minimal(curr_node->get_count());
    set_rule_node_pvalue(rule_node, label, curr_node->get_count());
    rule_node->set_min_log_p(1.0);

    if (rule_node->get_is_ss()) {
      vector<int> temp_items;           // to convert uint16_t to int
      vector<int> temp_original_items;  // to convert to original features
      for (auto x : curr_items) {
        temp_items.push_back(x);
        temp_original_items.push_back(this->feature_rmap[x]);
      }
      Rule *r =
          new Rule((double)curr_node->get_count() / this->d_size,
                   (double)rule_node->get_count() / curr_node->get_count(),
                   exp(rule_node->get_log_p()), rule_node->get_log_p(), label,
                   temp_items, this->class_rmap.at(label), temp_original_items);
      this->initial_rules[label].push_back(r);
    }
    if (rule_node->get_is_minimal()) {
      curr_node->remove_label(label);
      continue;
    }
  }
}

void CPPSigDirect::set_pvalues_node_2(const int &curr_depth, Node *curr_node,
                                      const vector<uint16_t> &curr_items) {
  /* for leaves in the second layer and onward */
  auto p_temp = curr_node->get_rule_nodes();
  int label;
  RuleNode *rule_node;
  double min_log_p;
  bool parents_pss_not_minimal;
  pair<bool, double> pr;
  curr_node->shrink_vectors();
  for (size_t i = 0; i < this->class_map.size(); i++) {
    if (p_temp[i] == NULL) continue;
    label = i;
    rule_node = p_temp[i];

    are_parents_pss_and_not_minimal_and_min_pss(curr_items, label, pr);
    parents_pss_not_minimal = pr.first;
    min_log_p = pr.second;

    rule_node->set_is_minimal(curr_node->get_count());

    if (!parents_pss_not_minimal) {
      rule_node->set_min_log_p(min_log_p);

      if (rule_node->get_is_minimal()) {
        curr_node->remove_label(label);
      }
      continue;
    }
    set_rule_node_pvalue(rule_node, label, curr_node->get_count());
    if (!rule_node->get_is_pss()) {  // do not remove the label here!
    } else if (rule_node->get_is_ss() && rule_node->get_log_p() < min_log_p) {
      vector<int> temp_items;           // to convert uint16_t to int
      vector<int> temp_original_items;  // to convert to original features
      for (auto x : curr_items) {
        temp_items.push_back(x);
        temp_original_items.push_back(this->feature_rmap[x]);
      }
      Rule *r =
          new Rule((double)curr_node->get_count() / this->d_size,
                   (double)rule_node->get_count() / curr_node->get_count(),
                   exp(rule_node->get_log_p()), rule_node->get_log_p(), label,
                   temp_items, this->class_rmap.at(label), temp_original_items);
      this->initial_rules[label].push_back(r);
    }
    if (rule_node->get_is_minimal()) {
      curr_node->remove_label(label);
      continue;
    }
    if (curr_node->has_label(label)) {
      rule_node->set_min_log_p(min_log_p);
    }
  }
}

void CPPSigDirect::set_pvalues(const int &max_depth, int curr_depth,
                               Node *curr_node, vector<uint16_t> &curr_items) {
  if (curr_depth == max_depth && max_depth == 1) {
    this->set_pvalues_node_1(curr_depth, curr_node, curr_items);
    return;
  }
  if (curr_depth == max_depth && max_depth != 1) {
    this->set_pvalues_node_2(curr_depth, curr_node, curr_items);
    return;
  }
  for (auto item_nodes : curr_node->get_children()) {
    curr_items[curr_depth] = item_nodes;
    set_pvalues(max_depth, curr_depth + 1, curr_node->get_child(item_nodes),
                curr_items);
  }
}

void CPPSigDirect::set_rule_node_pvalue(RuleNode *rule_node, const int &label,
                                        const int &antecedent_support) {
  int label_support = get_label_support(label);
  double labels_support_lgamma = get_label_support_lgamma(label);
  int subsequent_support = rule_node->get_count();
  rule_node->set_pss(antecedent_support, label_support, this->d_size,
                     this->d_size_lgamma, labels_support_lgamma);
  if (rule_node->get_is_pss())
    rule_node->set_p(antecedent_support, label_support, this->d_size,
                     subsequent_support);
}

void CPPSigDirect::are_parents_pss_and_not_minimal_and_min_pss(
    const vector<uint16_t> &items, const int &label, pair<bool, double> &ret) {
  pair<bool, double> pa;
  ret.second = 1.0;
  for (size_t ignore_index = 0; ignore_index < items.size(); ignore_index++) {
    is_rule_pss_and_not_minimal(items, label, ignore_index,
                                pa);  // <pss_not_minimal, log_pvalue>
    ret.second = min(ret.second, pa.second);
    if (!pa.first) {  // pss_not_minimal
      ret.first = false;
      return;  // pair(false, min_all_pvalue);
    }
  }
  ret.first = true;
  return;  // pair(true, min_all_pvalue);
}

void CPPSigDirect::is_rule_pss_and_not_minimal(const vector<uint16_t> &items,
                                               const int &label,
                                               const uint16_t &ignore_index,
                                               pair<bool, double> &ret) {
  /* returns a pair (pass by ref last arg):
   * 1. is_rule_pss_and_not_minimal,
   * 2. min_log_p*/
  Node *curr_node = this->root;
  Node *next_node;
  double min_log_p = 1.0;
  ret.first = false;
  ret.second = 0.0;
  for (size_t i = 0; i < items.size(); i++) {
    if (i == ignore_index) continue;
    next_node = curr_node->get_child(items[i]);
    if (next_node == NULL) return;  // pair(false, 0.0);
    curr_node = next_node;

    if (!curr_node->has_label(label)) return;  // pair(false, 0.0);
    RuleNode *rule_node = curr_node->get_rule_node(label);
    if (!rule_node->get_is_pss()) {
      return;  // pair(false, 0.0);
    }
    if (rule_node->get_is_minimal()) {
      return;  // pair(false, 0.0);
    }
  }
  if (!curr_node->has_label(label)) return;  // pair<bool, double>(false, 0.0);
  RuleNode *rule_node = curr_node->get_rule_node(label);
  min_log_p = min(min_log_p, rule_node->get_min_log_p());
  if (rule_node->get_is_pss() && !rule_node->get_is_minimal()) {
    ret.first = true;
    ret.second = min_log_p;
    return;  // pair(true, min_log_p);
  }
  return;  // pair(false, 0.0);
}

/* Prune rules */
void CPPSigDirect::update_confidence_prune(
    vector<unordered_set<int>> prune_data_set, vector<int> prune_data_classes,
    const vector<bool> &instance_validity, vector<Rule *> input_rules) {
  for (uint16_t r_index = 0; r_index < input_rules.size(); r_index++) {
    int match_count = 0, diff_count = 0;
    input_rules[r_index]->set_confidence_prune(0.0);
    for (uint16_t i_index = 0; i_index < prune_data_set.size(); i_index++) {
      if (instance_validity[i_index] == false) continue;
      bool is_match = true;
      for (int x : input_rules[r_index]->get_items())
        if (prune_data_set[i_index].find(x) == prune_data_set[i_index].end()) {
          is_match = false;
          break;
        }
      if (is_match) {
        if (input_rules[r_index]->get_label() == prune_data_classes[i_index])
          match_count++;
        else
          diff_count++;
      }
    }
    if (match_count + diff_count == 0) {
      input_rules[r_index]->set_confidence_prune(0.0);
    } else {
      input_rules[r_index]->set_confidence_prune((double)match_count /
                                                 (match_count + diff_count));
    }
  }
}

void CPPSigDirect::prune_rules_sigd2(
    vector<vector<int>> prune_data, vector<int> prune_data_classes,
    unordered_map<int, vector<Rule *>> input_rules) {
  unordered_map<int, vector<Rule *>> intermediate_rules;
  // compute confidence scores based on prune set.
  vector<Rule *> valid_rules;
  for (auto p : input_rules) {
    for (auto r : p.second) {
      valid_rules.push_back(r);
    }
  }
  sort(valid_rules.begin(), valid_rules.end(), [](auto a, auto b) {
    return a->get_confidence() < b->get_confidence();
  });
  vector<unordered_set<int>> prune_data_set(prune_data.size());
  for (uint16_t index = 0; index < prune_data.size(); index++)
    prune_data_set[index] =
        unordered_set<int>(prune_data[index].begin(), prune_data[index].end());
  vector<bool> instance_validity(prune_data.size(), true);

  // first step of pruning
  while (valid_rules.size() > 0 &&
         find(instance_validity.begin(), instance_validity.end(), true) !=
             instance_validity.end()) {
    int index =
        distance(valid_rules.begin(),
                 max_element(valid_rules.begin(), valid_rules.end(),
                             [](const Rule *a, const Rule *b) {
                               return a->get_confidence_prune() !=
                                              b->get_confidence_prune()
                                          ? a->get_confidence_prune() <
                                                b->get_confidence_prune()
                                          : a->get_log_p() > b->get_log_p();
                             }));
    int label = valid_rules[index]->get_label();
    if (valid_rules[index]->get_confidence_prune() <
        config.confidence_threshold)
      break;
    for (uint16_t i_index = 0; i_index < prune_data_set.size(); i_index++) {
      if (instance_validity[i_index] == false) continue;
      if (valid_rules[index]->applies_to(prune_data_set[i_index])) {
        instance_validity[i_index] = false;
      }
    }
    intermediate_rules[label].push_back(valid_rules[index]);
    valid_rules.erase(valid_rules.begin() + index);
    update_confidence_prune(prune_data_set, prune_data_classes,
                            instance_validity, valid_rules);
  }
  instance_validity = vector<bool>(prune_data.size(), true);
  valid_rules.clear();
  for (auto p : intermediate_rules) {
    for (auto r : p.second) {
      valid_rules.push_back(r);
    }
  }
  update_confidence_prune(prune_data_set, prune_data_classes, instance_validity,
                          valid_rules);
  // original SigDirect's pruning
  prune_rules_sd(prune_data, prune_data_classes, intermediate_rules);
}

void CPPSigDirect::prune_rules_sd(
    vector<vector<int>> prune_data, vector<int> prune_data_classes,
    unordered_map<int, vector<Rule *>> input_rules) {
  for (size_t i = 0; i < prune_data.size(); i++) {
    unordered_set<int> items_set(prune_data[i].begin(), prune_data[i].end());
    vector<Rule *> applicable_rules =
        get_applicable_rules(items_set, input_rules[prune_data_classes[i]]);
    if (applicable_rules.size() == 0) {  // no applicable rule!
      continue;
    }
    Rule *best_rule;
    best_rule = *max_element(
        applicable_rules.begin(), applicable_rules.end(),
        [](const auto &lhs, const auto &rhs) {
          return lhs->get_confidence() != rhs->get_confidence()
                     ? lhs->get_confidence() < rhs->get_confidence()
                     : lhs->get_support() < rhs->get_support();
        });

    best_rule->inc_importance();
  }
  for (auto p : this->initial_rules) {
    for (auto r : p.second) {
      if (r->get_importance() > 0) this->final_rules[p.first].push_back(r);
    }
  }
}

/* Prediction */
vector<int> CPPSigDirect::predict(const vector<vector<int>> &X,
                                  const Heuristic &hrs) {
  // preprocess X
  vector<vector<int>> processed_X = transform_data(X);

  vector<int> final_labels(processed_X.size(), 0);
  for (size_t i = 0; i < processed_X.size(); i++) {
    unordered_set<int> items_set(processed_X[i].begin(), processed_X[i].end());
    final_labels[i] = predict_instance(items_set, hrs);
  }
  return untransform_labels(final_labels);
}

int CPPSigDirect::predict_instance(unordered_set<int> items_set,
                                   Heuristic hrs) {
  vector<pair<int, int>> scores;
  unordered_map<int, vector<Rule *>> applicable_rule_labels;
  for (auto p : this->final_rules) {
    int label = p.first;
    auto rules = p.second;
    applicable_rule_labels[label] = get_applicable_rules(items_set, rules);
  }

  if (hrs == Heuristic::hrs1) {
    for (auto l : applicable_rule_labels) {
      long double score = 0.0;
      int label = l.first;
      for (auto rule : l.second) {
        score += rule->get_log_p() * (double)rule->get_importance();
      }
      scores.push_back(pair<int, long double>(label, score));
    }
    auto result = min_element(
        scores.begin(), scores.end(), [this](const auto &lhs, const auto &rhs) {
          return lhs.second != rhs.second ? lhs.second < rhs.second
                                          : lhs.first == this->majority_class;
        });
    if (scores.size() > 0) return result->first;
  } else if (hrs == Heuristic::hrs2) {
    for (auto l : applicable_rule_labels) {
      long double score = 0.0;
      int label = l.first;
      for (auto rule : l.second)
        score += rule->get_confidence() * (double)rule->get_importance();
      scores.push_back(pair<int, long double>(label, score));
    }
    auto result = max_element(
        scores.begin(), scores.end(), [this](const auto &lhs, const auto &rhs) {
          return lhs.second != rhs.second ? lhs.second < rhs.second
                                          : lhs.first != this->majority_class;
        });
    if (scores.size() > 0) return result->first;
  } else if (hrs == Heuristic::hrs3) {
    for (auto l : applicable_rule_labels) {
      long double score = 0.0;
      int label = l.first;
      for (auto rule : l.second)
        score += (double)rule->get_importance() * rule->get_log_p() *
                 rule->get_confidence();
      scores.push_back(pair<int, long double>(label, score));
    }
    auto result = min_element(
        scores.begin(), scores.end(), [this](const auto &lhs, const auto &rhs) {
          return lhs.second != rhs.second ? lhs.second < rhs.second
                                          : lhs.first == this->majority_class;
        });
    if (scores.size() > 0) return result->first;
  } else {
    throw "wrong heuristic provided!";
  }
  PLOG_WARNING << "majority class selected";
  return this->majority_class;
}

vector<Rule *> CPPSigDirect::get_applicable_rules(unordered_set<int> items_set,
                                                  vector<Rule *> rules) {
  vector<Rule *> applicables;
  for (auto rule : rules) {
    if (rule->applies_to(items_set)) {
      applicables.push_back(rule);
    }
  }
  return applicables;
}

unordered_map<int, vector<Rule>> CPPSigDirect::get_all_rules() {
  unordered_map<int, vector<Rule>> ret;
  for (auto p : this->final_rules) {
    for (auto rule_ptr : p.second) {
      Rule r = *rule_ptr;
      ret[this->class_rmap[p.first]].push_back(r);
    }
  }
  return ret;
}
