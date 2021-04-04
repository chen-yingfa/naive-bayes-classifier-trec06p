cd src

ip_weights=(
    0
    0.01
    0.1
    1
    2
    5
    10
)
time_weights=(
    0
    0.01 
    0.1  
    1 
    2 
    5
    10
)

for ip_weight in ${ip_weights[@]} 
do 
    for time_weight in ${time_weights[@]}
    do
        echo "Testing ip weight = ${ip_weight}, time weight = ${time_weight}"
        python3 test.py \
        --use_ip \
        --use_time\
        --time_weight $time_weight \
        --ip_weight $ip_weight > \
        ../result/weight_${ip_weight}_${time_weight}.txt
    done 
done