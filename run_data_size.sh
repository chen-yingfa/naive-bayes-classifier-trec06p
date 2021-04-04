
sizes=(
    0.01 
    0.05 
    0.2 
    0.5 
    1.0
)

cd src
for size in ${sizes[@]}
do
    python3 preprocess.py --data_size ${size}
    python3 test.py > ../result/size_${size}.txt
done