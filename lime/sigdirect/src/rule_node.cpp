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
#include "rule_node.h"
using namespace std;

RuleNode::RuleNode() {
  this->count = 0;
  this->log_p = 0.0;
  this->min_log_p = 1.0;
  this->is_minimal = false;
}

RuleNode::~RuleNode() {}

double RuleNode::get_log_p() { return this->log_p; }

bool RuleNode::get_is_pss() { return this->is_pss; }

bool RuleNode::get_is_ss() { return this->log_p <= config.log_alpha; }

double RuleNode::get_min_log_p() { return this->min_log_p; }

int RuleNode::get_count() { return this->count; }

void RuleNode::set_count(uint16_t count) { this->count = count; }

void RuleNode::increase_count() { this->count++; }

bool RuleNode::get_is_minimal() { return this->is_minimal; }

void RuleNode::set_p(int antecedent_support, int label_support, int d_size,
                     int subsequent_support) {
  int min_n = min(antecedent_support, label_support) - subsequent_support;
  int intersection = subsequent_support;
  int union_ = (antecedent_support + label_support) - intersection;

  long double lz = lgamma(d_size + 1) - lgamma(label_support + 1) -
                   lgamma(d_size - label_support + 1);

  long double t1 = lgamma(antecedent_support + 1);
  long double t2 = lgamma(d_size - antecedent_support + 1);

  long double l1, l2, sum = 0.0;

  for (int i = 0; i < (min_n + 1); i++) {
    l1 = t1 - lgamma(subsequent_support + i + 1) -
         lgamma(antecedent_support - subsequent_support - i + 1);
    l2 = t2 - lgamma(d_size - union_ + i + 1) -
         lgamma(d_size - antecedent_support - d_size + union_ - i + 1);
    sum += exp(l1 + l2 - lz);
  }
  this->log_p = log(sum);
}

void RuleNode::set_pss(const int& antecedent_support, const int& label_support,
                       const int& d_size, const double& d_size_lgamma,
                       const double& label_support_lgamma) {
  if (antecedent_support > label_support) {
    this->is_pss = true;
  } else {
    double a0 = lgamma(d_size - antecedent_support + 1);
    double a1 = label_support_lgamma;
    double a2 = d_size_lgamma;
    double a3 = lgamma(label_support - antecedent_support + 1);

    this->is_pss = a0 + a1 - a2 - a3 <= config.log_alpha;
  }
}

void RuleNode::set_is_minimal(int antecedent_support) {
  if (this->get_count() >= antecedent_support)
    this->is_minimal = true;
  else
    this->is_minimal = false;
}

void RuleNode::set_min_log_p(double parent_log_p) {
  this->min_log_p = min(parent_log_p, this->log_p);
}
