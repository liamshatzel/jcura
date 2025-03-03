#!/bin/bash
ACCOUNT=brandonh
N=8
GPU=p100:1
REQ_TIME=1:00:00
MEM=32G
# salloc --account=def-someuser --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00
salloc --account=def-$ACCOUNT --cpus-per-task=$N --gres=gpu:$GPU --mem=$MEM --time=$REQ_TIME
