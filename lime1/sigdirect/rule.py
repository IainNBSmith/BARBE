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

import math

class Rule:
    """ Represents a final Rule that is extracted from tree. """

    def __init__(self, items, label, confidence, ss, support, importance=1, word_dict=None):
        self._items = items
        self._label = label
        self._confidence = confidence
        self._log_p = ss
        self._support = support
        self._importance = importance
        self._word_dict = word_dict

    def get_items(self):
        return self._items

    def get_label(self):
        return self._label

    def get_confidence(self):
        return self._confidence

    def get_log_p(self):
        return self._log_p

    def get_support(self):
        return self._support

    def set_items(self, new_items):
        self._items = new_items
    
    def get_importance(self):
        return self._importance
    
    def set_importance(self, importance):
        self._importance = float(importance)

    def __str__(self):
        try:
            return "{} {};{:.4f},{:.3f},{:.3f}".format(' '.join(map(self._word_dict.get, self._items)),
                                          self._label,
                                          self._support,
                                          self._confidence,
                                          self._log_p,
                                          )
        except Exception as e:
            print(repr(e), self._items, self._label, self._log_p)
            return "{} {};{:.4f},{:.3f},{:.3f}".format(' '.join(map(str, self._items)),
                                          self._label,
                                          float(self._support),
                                          float(self._confidence),
                                          0.0,
                                          )
