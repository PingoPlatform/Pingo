#!/bin/bash
for filename in /Users/peter/Work/boulderdetection/results/2020_05_22_redone_UHR_upscaling/valid_UHR_pad32_model60000/depad/*.pgw; do
    sed -i '' '1 s/^.*$/0.125/' $filename #change first line
    sed -i '' '4 s/^.*$/-0.125/' $filename #change 4th line
done