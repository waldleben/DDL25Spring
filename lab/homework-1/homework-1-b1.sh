#!/bin/bash
cd "$(dirname "$0")" || return
START_TIME=$SECONDS

for ((i=0; i<3; i=i+1))
do
    touch "out-b1-$i.txt"
    (sleep 1; python -u "homework-1-b1.py" $i>"out-b1-$i.txt") &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"