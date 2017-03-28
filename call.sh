min_size=10
case_id=1
for c in `seq 70 10 80`;
do
echo "====================================================================="
echo "Case $case_id:"
echo "c: $c"
echo "min_size: $min_size"
echo "----------------------------"
python main.py $c $min_size
case_id=`expr $case_id + 1`
done
