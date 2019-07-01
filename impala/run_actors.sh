#!/bin/bash

export FLAGS_fraction_of_gpu_memory_to_use=0.02

for i in $(seq 1 32); do
    export CUDA_VISIBLE_DEVICES="3"; export FLAGS_fraction_of_gpu_memory_to_use=0.02; python impala/actor.py &
done;
wait
