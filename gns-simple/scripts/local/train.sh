#!/bin/bash

python train.py \
    --data-path datasets/Sand \
    --output output \
    --epoch 10 \
    --eval-interval 100 \
    --vis-interval 100 \
    --save-interval 100

