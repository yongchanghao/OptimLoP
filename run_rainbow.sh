#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 rainbow.py \
--step-per-epoch=50000 \
--buffer-size=50000 \
--optimizer noiseadam \
--wandb-project lgq-atari
