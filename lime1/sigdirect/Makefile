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

SRC_DIR=/cshome/alamanik/barbetest/sigdirect/src/
LIB_DIR=/cshome/alamanik/barbetest/sigdirect/lib/plog/include/
INC_DIR=/cshome/alamanik/barbetest/sigdirect/include/
TESTS_DIR=/cshome/alamanik/barbetest/sigdirect/tests/
BIN_DIR=.

CC=g++-6.4
CFLAGS=-I$(SRC_DIR) -I$(LIB_DIR) -I$(INC_DIR) -I$(TESTS_DIR) -O3 -std=c++17 -MMD -Wall -DNDEBUG
LFLAGS=-std=c++11 -DNDEBUG

.PHONY: all clean

sigdirect_test: node.o rule.o rule_node.o sigdirect.o sigdirect_test.o
	rm -rf build/
	$(CC) $(LFLAGS) rule.o node.o rule_node.o sigdirect.o sigdirect_test.o -o sigdirect_test
	python setup.py build_ext --inplace
	@rm -f *.o

sigdirect_test.o: $(TESTS_DIR)sigdirect_test.cpp
	$(CC) $(CFLAGS) -c $(TESTS_DIR)sigdirect_test.cpp -o sigdirect_test.o

sigdirect.o: node.o rule.o rule_node.o $(SRC_DIR)sigdirect.cpp $(INC_DIR)sigdirect.h $(LIB_DIR)plog/Log.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)sigdirect.cpp -o sigdirect.o 

node.o: rule_node.o $(SRC_DIR)node.cpp $(SRC_DIR)node.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)node.cpp -o node.o

rule_node.o: $(SRC_DIR)rule_node.cpp $(SRC_DIR)rule_node.h $(SRC_DIR)config.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)rule_node.cpp -o rule_node.o

rule.o: $(SRC_DIR)rule.cpp $(SRC_DIR)rule.h
	$(CC) $(CFLAGS) -c $(SRC_DIR)rule.cpp -o rule.o

clean:
	@rm -f *.o
	@rm -f sigdirect_test

##### python setup.py  build_ext --inplace ####### 
