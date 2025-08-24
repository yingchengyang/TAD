#!/bin/bash

ALGO=$1
TASK=$2
GPU_ID=$3
EPISODES=$4
REWARD_SIZE=$5
TASK_WEIGHT=$6
TASK_LOSS_TYPE=$7
CONTEXT_DIM=$8

echo "Experiments started."
for seed in $(seq 0 4)
do
    xvfb-run -a python main.py --algo ${ALGO} --envs ${TASK} --action-repeat 2 --id ${ALGO} --device cuda:${GPU_ID} --seed $seed --episodes ${EPISODES} --sep-replay-buffer True --reward-size ${REWARD_SIZE} --task-weight ${TASK_WEIGHT} --task-loss-type ${TASK_LOSS_TYPE} --context-dim ${CONTEXT_DIM}
done
echo "Experiments ended."

# run command
# ./multitask_run.sh tad walker-stand~walker-walk 0 2000 10.0 0.1
# for ALGO, we can choose dreamer or planet or tad
# for TASK, we can choose
# for GPU_ID, we can choose 0~MAX_ID
# for EPISODES, we always choose 1000 or 2000

# ./multitask_run.sh planet p4echeetah-run~p4echeetah-run_back~p4echeetah-flip_forward~p4echeetah-flip_backward 0 2000 0.0 0.0