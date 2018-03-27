#!/bin/bash
# 1 track
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Down'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'

for i in `seq 1 0`
do
    xte 'key Right'
    xte 'usleep 100000'
done

xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'

#xte 'key F3'
#xte 'usleep 100000'
#xte 'key F3'
#xte 'usleep 100000'

xte 'key F2'
xte 'usleep 100000'
#xte 'key F2'
#xte 'usleep 100000'
#xte 'key F2'
#xte 'usleep 100000'
#xte 'key F2'
#xte 'usleep 100000'
#xte 'key F2'
#xte 'usleep 100000'