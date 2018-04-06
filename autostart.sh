#!/bin/bash

window=`xdotool search --name $1 | head -n 1`

xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Down
sleep 0.2
xdotool key --window $window Return
sleep 0.2

for i in `seq 1 $2`
do
    xdotool key --window $window Right
    sleep 0.2
done

xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Return
sleep 0.2

xdotool key --window $window F2