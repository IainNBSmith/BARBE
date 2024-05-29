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
#include "node.h"
using namespace std;

uint16_t Node::labels_size = 0;

Node::Node() {
  this->count = 0;
  this->rule_nodes = new RuleNode* [labels_size] { NULL };
}

Node::~Node() {
  for (auto p : this->children_nodes) {
    delete p;
  }
  for (int i = 0; i < labels_size; i++) {
    auto p = this->rule_nodes[i];
    if (p != NULL) delete p;
  }
  this->children_ids.clear();
  delete this->rule_nodes;
}

void Node::shrink_vectors() {
  this->children_ids.shrink_to_fit();
  this->children_nodes.shrink_to_fit();
}
uint16_t Node::get_count() { return this->count; }

void Node::add_count(uint16_t count) { this->count += count; }

void Node::increase_count() { this->count++; }

const vector<uint16_t>& Node::get_children() { return this->children_ids; }

Node* Node::get_child(const uint16_t& item) {
  auto it = find(this->children_ids.begin(), this->children_ids.end(), item);
  if (it == this->children_ids.end()) return NULL;
  return this->children_nodes[distance(this->children_ids.begin(), it)];
}

bool Node::has_child(const uint16_t& item) {
  return find(this->children_ids.begin(), this->children_ids.end(), item) !=
         this->children_ids.end();
}

void Node::add_child(uint16_t item, Node* node) {
  this->children_ids.push_back(item);
  this->children_nodes.push_back(node);
}

bool Node::has_label(uint16_t label) {
  if (this->rule_nodes[label] == NULL) return false;
  return true;
}

RuleNode* Node::add_rule_node(uint16_t label) {
  assert(this->rule_nodes[label] == NULL);
  this->rule_nodes[label] = new RuleNode();
  return this->rule_nodes[label];
}

RuleNode* Node::get_rule_node(uint16_t label) {
  assert(this->rule_nodes[label] != NULL);
  return this->rule_nodes[label];
}

uint16_t Node::get_labels_size() {
  uint16_t counter = 0;
  for (int i = 0; i < Node::labels_size; i++) {
    auto p = this->rule_nodes[i];
    if (p != NULL) counter++;
  }
  return counter;
}

RuleNode** Node::get_rule_nodes() { return this->rule_nodes; }

void Node::remove_label(uint16_t label) {
  if (this->rule_nodes[label] == NULL) {
    throw "label doesn't exist 2";
  }
  delete this->rule_nodes[label];
  this->rule_nodes[label] = NULL;
}
