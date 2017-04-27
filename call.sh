min_size=20
case_id=1
ds_name="hymap02_ds02"

for c in `seq 40 10 140`;
do
echo "====================================================================="
echo "Case $case_id:"
echo "c: $c"
echo "min_size: $min_size"
echo "----------------------------"
python main.py $c $min_size $ds_name
case_id=`expr $case_id + 1`
done
