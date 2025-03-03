#!/bin/bash

python train.py \
    --data-path datasets/WaterDropSample \
    --output output \
    --epoch 1 \
    --eval-interval 100 \
    --vis-interval 100 \
    --save-interval 100

