cd src

smooth=(
    0.0000000000000001
    0.00000001
    0.0001
    0.01
    0.1
    1
    10
    100
)
for s in ${smooth[@]}
do
    python3 test.py --smooth $s > ../result/smooth_${s}.txt
done