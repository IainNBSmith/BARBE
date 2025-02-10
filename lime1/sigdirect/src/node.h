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
#ifndef _NODE
#define _NODE

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "rule_node.h"
using namespace std;

class Node {
 private:
  vector<uint16_t> children_ids;
  vector<Node*> children_nodes;

  RuleNode** rule_nodes;
  uint16_t count;

 public:
  static uint16_t labels_size;

  Node();
  ~Node();

  uint16_t get_count();
  void add_count(uint16_t);
  void increase_count();
  const vector<uint16_t>& get_children();
  Node* get_child(const uint16_t&);
  bool has_child(const uint16_t&);
  void add_child(uint16_t, Node*);

  bool has_label(uint16_t);
  RuleNode* add_rule_node(uint16_t);
  RuleNode* get_rule_node(uint16_t);

  uint16_t get_labels_size();
  RuleNode** get_rule_nodes();
  void remove_label(uint16_t);
  void shrink_vectors();
};

#endif
