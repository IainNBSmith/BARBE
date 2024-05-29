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

import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append('../')
import sigdirect

binarize = True
UCI_PATH = '../uci/'

class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the form used in sklearn datasets  
            (to follow sklearn method, X will always be 2d np matrix)"""
        transaction_data =  [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X,y = [], []

        for transaction in transaction_data:
            if binarize:
                positions = np.array(transaction[:-1]) - 1
                transaction_np = np.zeros((max_val))
                transaction_np[positions] = 1
            else:
                transaction_np = transaction[:-1]
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        return X,y

def test_uci():

    assert len(sys.argv)>1
    dataset_name = sys.argv[1]
    print("DATASET:", dataset_name)

    if len(sys.argv)>2:
        start_index = int(sys.argv[2])
    else:
        start_index = 1
    
    final_index = 10
    k = final_index - start_index + 1

    all_pred_y = defaultdict(list)
    all_true_y = []

    # counting number of rules before and after pruning
    generated_counter = 0
    final_counter     = 0
    avg = [0.0] * 4

    tt1 = time.time()

    for index in range(start_index, final_index +1):

        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = os.path.join(UCI_PATH, '{}_tr{}.txt'.format(dataset_name, index))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')
        X,y = prep.preprocess_data(raw_data)

        clf = sigdirect.SigDirect(clf_version=1, 
                                    early_stopping=True, 
                                    alpha=0.05, 
                                    is_binary=True, 
                                    get_logs=sys.stdout)
        generated_c, final_c = clf.fit(X, y)
        
        generated_counter += generated_c
        final_counter     += final_c

        # load the test data and pre-process it.
        test_filename  = os.path.join(UCI_PATH, '{}_ts{}.txt'.format(dataset_name, index))
        with open(test_filename) as f:
            raw_test_data = f.read().strip().split('\n')
        X,y = prep.preprocess_data(raw_test_data)

        # evaluate the classifier using different heuristics for pruning
        for hrs in (1,2,3):
            y_pred = clf.predict(X, hrs)
            avg[hrs] += accuracy_score(y, y_pred)

            all_pred_y[hrs].extend(y_pred)

        all_true_y.extend(list(y))

    print(dataset_name)
    for hrs in (1,2,3):
        print('AVG ACC S{}:'.format(hrs), accuracy_score(all_true_y, all_pred_y[hrs]))
    print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter/k, final_counter/k))
    print('TOTAL TIME:', time.time()-tt1)
    
if __name__ == '__main__':
    test_uci()
