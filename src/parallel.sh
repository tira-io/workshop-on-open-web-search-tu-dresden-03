#!/bin/bash

### TOP LEVEL SCRIPT
### Parallelizes on dataset-level by calling main.py with the number of GPUs
###   available and the worker number, e.g. `python3 main.py 2 4` to work on
###   datasets slice 2/4 (and distribute across 4 GPUs).
###
### main.py may also be run directly, maybe for ChatGPT, via
###   `python3 main.py 1 1`

numgpus=$(python3 gpucount.py)

for i in $(seq 1 $numgpus); do
    CUDA_VISIBLE_DEVICES=$(( $i-1 )) python3 main.py $i $numgpus &
done
wait
