#!/bin/bash

CUDA_VISIBLE_DEVICES="0" \
torchrun --standalone \
         --nproc_per_node=gpu \
         pretrain_distributed.py
