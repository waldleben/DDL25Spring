for ((i=0; i<3; i=i+1))
do

    touch "out$i.txt"
    
    (sleep 1; python -u "intro_PP_1F1B.py" $i>"out$i.txt") &


done