#!/bin/bash

mkdir -p logs
NUM_TRIALS=5
LRS=(0.000017 0.000019 0.000070 0.000102 0.000199)
TRAINING_STEPS=(181 128 175 129 191)
WARMUP_STEPS=(62 22 55 50 56)
WEIGHT_DECAYS=(0.22246514992794986 0.030748552851452247 0.008939165831421103 0.007960790905159087 0.163482444180965)


for ((i=1; i<=NUM_TRIALS; i++)); do
# python main_dis
    python main_distillation.py distill-task1-olmo \
        --lr ${LRS[$i-1]} \
        --training_steps ${TRAINING_STEPS[$i-1]} \
        --warmup_steps ${WARMUP_STEPS[$i-1]} \
        --weight_decay ${WEIGHT_DECAYS[$i-1]} \
done