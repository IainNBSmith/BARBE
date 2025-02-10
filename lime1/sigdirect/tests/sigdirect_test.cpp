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
#include "sigdirect.h"

#include <climits>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

string UCI_PATH = "uci/";
char delim = ' ';

vector<vector<int>> load_raw_data(const string& filename, const char& delim) {
  /* read the provided file, and return a 2d vector of integers */
  ifstream infile;
  infile.open(filename, ios::in);
  if (!infile) {
    cout << "could not open training data file: " << filename << endl;
    throw "FileNotFound";
  }

  string s;
  vector<vector<int>> data;

  while (getline(infile, s)) {
    vector<int> instance;
    stringstream ss(s);
    string token;
    while (std::getline(ss, token, delim)) {
      instance.push_back(stoi(token));
    }
    data.push_back(instance);
  }

  return data;
}

void separate_labels(const vector<vector<int>> data, vector<vector<int>>& X,
                     vector<int>& y) {
  for (auto p : data) {
    X.push_back(std::vector<int>(p.begin(), p.end() - 1));
    y.push_back(p[p.size() - 1]);
  }
}

float evaluate(const vector<int>& y_test, const vector<int>& y_predicted) {
  assert(y_test.size() == y_predicted.size());
  assert(y_test.size() > 0);

  int true_count = 0;
  for (size_t i = 0; i < y_test.size(); i++) {
    if (y_test[i] == y_predicted[i]) true_count++;
  }
  return (float)true_count / y_test.size();
}

int main(int argc, char* argv[]) {
  /* Input example: ./sigdirect_test iris
   * It will run iris dataset (from fold 1 to 10) and output avg results.
   * Alternatively, a number could be passes as an additional arg,
   * to run from that fold onward: ./sigdirect_test iris 10 (only 10th fold)
   * uncomment commented lines, to print info for each fold. */
  uint16_t start_fold = 1;
  uint16_t end_fold = 10;
  if (argc < 2) {
    cout << "filename not provided! terminating..." << endl;
    return 1;
  } else if (argc > 4) {
    cout << "too many arguments! terminating..." << endl;
    return 1;
  } else if (argc == 3) {
    start_fold = stoi(argv[2]);
  } else if (argc == 4) {
    start_fold = stoi(argv[2]);
    end_fold = stoi(argv[3]);
  }

  clock_t begin = clock();
  int total_initial_rule_count = 0;
  int total_final_rule_count = 0;
  cout << argv[1] << endl;
  double results_hrs1 = 0.0, results_hrs2 = 0.0, results_hrs3 = 0.0;

  for (size_t fold_number = start_fold; fold_number <= end_fold;
       fold_number++) {
    clock_t fold_begin = clock();
    vector<vector<int>> X_train, X_test;
    vector<int> y_train, y_test, y_predicted;

    // load train data
    string filename_train =
        UCI_PATH + string(argv[1]) + "_tr" + to_string(fold_number) + ".txt";
    vector<vector<int>> data_train = load_raw_data(filename_train, delim);
    separate_labels(data_train, X_train, y_train);

    // load test data
    string filename_test =
        UCI_PATH + string(argv[1]) + "_ts" + to_string(fold_number) + ".txt";
    vector<vector<int>> data_test = load_raw_data(filename_test, delim);
    separate_labels(data_test, X_test, y_test);

    CPPSigDirect* sd;
    try {
      sd = new CPPSigDirect(1, 0.05, true, 0.5);
      //      cout << filename << " data.size(): " << data.size() << "
      //      max_element+1:" << max_element+1 << endl; total_rule_count +=
      //      sd.fit(raw_data, classes, data.size(), max_element+1);
      auto p = sd->fit(X_train, y_train);
      total_initial_rule_count += p.first;
      total_final_rule_count += p.second;
      y_predicted = sd->predict(X_test, Heuristic::hrs1);
      results_hrs1 += evaluate(y_test, y_predicted);
      //      cout << "hrs1: " << evaluate(y_test, y_predicted) << endl;
      y_predicted = sd->predict(X_test, Heuristic::hrs2);
      results_hrs2 += evaluate(y_test, y_predicted);
      //      cout << "hrs2: " << evaluate(y_test, y_predicted) << endl;
      y_predicted = sd->predict(X_test, Heuristic::hrs3);
      results_hrs3 += evaluate(y_test, y_predicted);
      //      cout << "hrs3: " << evaluate(y_test, y_predicted) << endl <<
      //      endl;
    } catch (exception& e) {
      cout << "Exception: " << e.what() << endl;
      throw e;
    }
    clock_t fold_end = clock();
    double fold_elapsed_secs = double(fold_end - fold_begin) / CLOCKS_PER_SEC;
    if (false) cout << "Fold TIME: " << fold_elapsed_secs << endl << endl;
    delete sd;
  }
  cout << "hrs1: " << results_hrs1 / (end_fold - start_fold + 1) << endl;
  cout << "hrs2: " << results_hrs2 / (end_fold - start_fold + 1) << endl;
  cout << "hrs3: " << results_hrs3 / (end_fold - start_fold + 1) << endl;
  cout << "Average RULES: "
       << (float)total_initial_rule_count / (end_fold - start_fold + 1) << " "
       << (float)total_final_rule_count / (end_fold - start_fold + 1) << endl;
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "TIME: " << elapsed_secs << endl << endl;
  return 0;
}
