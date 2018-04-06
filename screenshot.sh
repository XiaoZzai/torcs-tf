#!/bin/bash

window=`xdotool search --name $1 | head -n 1`

import -window $window .tmp.png