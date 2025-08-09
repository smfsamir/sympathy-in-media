#!/bin/bash
#SBATCH --qos=a100_smfsamir
#SBATCH --time=12:00:00
#SBATCH --job-name=meta-ft
#SBATCH --gres=gpu:a100:1 
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca
#SBATCH --output=R-olmo-%x.%j.out
#SBATCH --error=R-olmo-%x.%j.err
source /h/smfsamir/my-venv/bin/activate
cd sympathy-in-media

mkdir -p logs
NUM_TRIALS=5
LRS=(0.000017 0.000019 0.000070 0.000102 0.000199)
TRAINING_STEPS=(181 128 175 129 191)
WARMUP_STEPS=(62 22 55 50 56)
WEIGHT_DECAYS=(0.22246514992794986 0.030748552851452247 0.008939165831421103 0.007960790905159087 0.163482444180965)


for ((i=1; i<=NUM_TRIALS; i++)); do
    echo "Running with:"
    echo "  learning_rate=${LRS[$i-1]}"
    echo "  weight_decay=${WEIGHT_DECAYS[$i-1]}"
    echo "  training_steps=${TRAINING_STEPS[$i-1]}"
    echo "  warmup_steps=${WARMUP_STEPS[$i-1]}"

    logfile="logs/lr=${LRS[$i-1]}_wd=${WEIGHT_DECAYS[$i-1]}_ts=${TRAINING_STEPS[$i-1]}_ws=${WARMUP_STEPS[$i-1]}.err"

    python main_distillation.py distill-task1-olmo \
        --learning_rate ${LRS[$i-1]} \
        --num_training_steps ${TRAINING_STEPS[$i-1]} \
        --warmup_steps ${WARMUP_STEPS[$i-1]} \
        --weight_decay ${WEIGHT_DECAYS[$i-1]} 2> "$logfile" \
        --model_name "meta"
done