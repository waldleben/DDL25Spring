#!/bin/bash
cd "$(dirname "$0")" || return
START_TIME=$SECONDS

for ((i=0; i<6; i=i+1))
do
    touch "out-b2-$i.txt"
    (sleep 1; python -u "homework-1-b2.py" $i>"out-b2-$i.txt") &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"