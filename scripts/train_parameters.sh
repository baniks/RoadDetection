#!/bin/bash
#######################################################################
#   File name: train_parameters.sh
#   Author: Soubarna Banik
#   Description: invokes main.py for series of parameters
#######################################################################

min_size=10
case_id=1
ds_name="hymap02_ds02"

for k in `seq 40 10 140`;
do
echo "====================================================================="
echo "Case $case_id:"
echo "k: $k"
echo "min_size: $min_size"
echo "----------------------------"
python main.py $k $min_size $ds_name
case_id=`expr $case_id + 1`
done
