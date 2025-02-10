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
#ifndef _RULE_NODE
#define _RULE_NODE

#include <math.h>

#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include "config.h"

using namespace std;

extern struct numbers config;

class RuleNode {
 private:
  double log_p;      // logarithm of p-value
  double min_log_p;  // minimum of log of p-value of parents
  uint16_t count;
  bool is_minimal;  // if support of X == support of (X,Y)
  bool is_pss;      // is potentially statistically significant

 public:
  RuleNode();
  ~RuleNode();

  double get_log_p();
  bool get_is_pss();
  bool get_is_ss();
  double get_min_log_p();
  int get_count();
  void set_count(uint16_t);
  void increase_count();
  bool get_is_minimal();

  void set_p(int, int, int, int);
  void set_pss(const int&, const int&, const int&, const double&,
               const double&);
  void set_is_minimal(int);
  void set_min_log_p(double);
};

#endif
