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
#include "rule.h"
using namespace std;

Rule::Rule(double support, long double confidence, long double p,
           long double log_p, int label, vector<int> items, int original_label,
           vector<int> original_items) {
  this->support = support;
  this->confidence = confidence;
  this->p = p;
  this->log_p = log_p;
  this->label = label;
  this->original_label = original_label;
  this->items = items;
  this->original_items = original_items;
  this->items_set = unordered_set<int>(items.begin(), items.end());
  this->importance = 0;
  this->confidence_prune = confidence;
}

Rule::Rule(const Rule& r) {
  this->support = r.support;
  this->confidence = r.confidence;
  this->p = r.p;
  this->log_p = r.log_p;

  this->label = r.label;
  this->original_label = r.original_label;
  this->items = r.items;
  this->original_items = r.original_items;

  this->items_set = unordered_set<int>(r.items.begin(), r.items.end());
  this->importance = r.importance;
  this->confidence_prune = r.confidence;
}

Rule::~Rule() {}

vector<int> Rule::get_items() const { return this->items; }

vector<int> Rule::get_original_items() const { return this->original_items; }

int Rule::get_label() const { return this->label; }

int Rule::get_original_label() const { return this->original_label; }

double Rule::get_support() const { return this->support; }

double long Rule::get_confidence() const { return this->confidence; }

double long Rule::get_p() const { return this->p; }

double long Rule::get_log_p() const { return this->log_p; }

void Rule::set_importance(int importance) { this->importance = importance; }

void Rule::inc_importance() { this->importance++; }

int Rule::get_importance() const { return this->importance; }

bool Rule::applies_to(unordered_set<int> items_set) {
  for (auto p : this->items) {
    if (items_set.find(p) == items_set.end()) return false;
  }
  return true;
}

void Rule::set_confidence_prune(long double c) { this->confidence_prune = c; }

long double Rule::get_confidence_prune() const {
  return this->confidence_prune;
}

void Rule::print_rule(unordered_map<int, int>& feature_rmap,
                      unordered_map<int, int>& class_rmap, ofstream& fout) {
  for (auto x : this->items) fout << feature_rmap[x] << " ";
  fout << "--> " << class_rmap[this->label] << " || ";
  fout << get_support() << " " << get_confidence() << " " << get_p() << " "
       << get_importance() << endl;
}

void Rule::print_raw_rule() {
  for (auto x : this->items) cout << x << " ";
  cout << "--> " << this->label << " || ";
  cout << get_support() << " " << get_confidence() << " " << get_p() << " "
       << get_importance() << endl;
}
