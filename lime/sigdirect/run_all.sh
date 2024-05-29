#!/bin/bash

INDEX=22
PY_INPTERPRETER=python
declare -a NAMES=( "breast" "anneal" "flare" "glass" "heart" "hepati" "horse" "iris" "led7" "pageBlocks" "pima" "wine" "zoo" "mushroom" "adult" "penDigits" "soybean" "ionosphere" "letRecog" "cylBands" )

cd tests/
for NAME in ${NAMES[@]};
do
	echo $NAME
	# ./sigdirect_test $NAME >> output_cpp_$INDEX
	$PY_INPTERPRETER sigdirect_test.py $NAME 10  >> output_py_$INDEX;
	echo "done"
done
