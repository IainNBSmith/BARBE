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
#ifndef _RULE
#define _RULE

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

class Rule {
 private:
  vector<int> items;           // based on SD features
  vector<int> original_items;  // based on feature ids provided by user
  unordered_set<int> items_set;
  int label;           // SD's internal label
  int original_label;  // based on label provided by user
  double support;
  double long confidence;
  double long confidence_prune;

  double long p;
  double long log_p;
  int importance;  // used when labelling an instance
 public:
  Rule(double, long double, long double, long double, int, vector<int>, int,
       vector<int>);
  Rule(const Rule&);
  ~Rule();

  vector<int> get_items() const;
  vector<int> get_original_items() const;
  int get_label() const;
  int get_original_label() const;
  double get_support() const;
  long double get_confidence() const;
  long double get_p() const;
  long double get_log_p() const;
  bool applies_to(unordered_set<int>);

  void set_confidence_prune(long double);
  long double get_confidence_prune() const;

  void set_importance(int);
  void inc_importance();
  int get_importance() const;

  void print_rule(unordered_map<int, int>&, unordered_map<int, int>&,
                  ofstream&);
  void print_raw_rule();
};

#endif
