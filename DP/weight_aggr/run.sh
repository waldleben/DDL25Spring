for ((i=0; i<3; i=i+1))
do

    touch "out$i.txt"
    
    (sleep 1; python -u "intro_DP_WA.py" $i>"out$i.txt") &


done