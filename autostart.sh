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

if (( $2 > 0 ))
then
    for i in `seq 1 $2`
    do
        xdotool key --window $window Right
        sleep 0.2
    done
else
    (( res=0-$2 ))
    for i in `seq 1 $res`
    do
        xdotool key --window $window Left
        sleep 0.2
    done
fi

xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Return
sleep 0.2
xdotool key --window $window Up
sleep 0.2
xdotool key --window $window Return
sleep 0.2

xdotool key --window $window F2