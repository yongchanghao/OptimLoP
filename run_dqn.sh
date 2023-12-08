#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 dqn.py \
 --step-per-epoch=50000 \
 --buffer-size=50000 \
 --optimizer lion \
 --wandb-project lgq-atari