#!/bin/bash

ALGO=$1
CFG=$2
GPU_ID=$3
REWARD_SIZE=$4
TASK_WEIGHT=$5

echo "Experiments started."
for seed in $(seq 0 4)
do
    xvfb-run -a python main.py --algo ${ALGO} --cfg ${CFG} --id ${ALGO} --device cuda:${GPU_ID} --seed $seed --sep-replay-buffer True --reward-size ${REWARD_SIZE} --task-weight ${TASK_WEIGHT}
done
echo "Experiments ended."

# run command
# ./mujoco_run.sh planet cheetah-dir 0 0.0 0.0
# ./mujoco_run.sh dreamer cheetah-dir 0 0.0 0.0
# ./mujoco_run.sh tad cheetah-dir1 0 1.0 0.1
# ./mujoco_run.sh tad cheetah-dir1 0 0.1 0.1

# ./mujoco_run.sh planet cheetah-vel 0 0.0 0.0
# ./mujoco_run.sh dreamer cheetah-vel 0 0.0 0.0
# ./mujoco_run.sh tad cheetah-vel1 0 0.1 0.5
# ./mujoco_run.sh tad cheetah-vel1 0 0.1 0.1