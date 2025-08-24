#!/bin/bash

ALGO=$1
TASK=$2
TEST_TASK=$3
GPU_ID=$4
EPISODES=$5
REWARD_SIZE=$6
TASK_WEIGHT=$7
TASK_LOSS_TYPE=$8
CONTEXT_DIM=$9

echo "Experiments started."
for seed in $(seq 0 4)
do
    xvfb-run -a python main.py --test-interval 50 --algo ${ALGO} --envs ${TASK} --test-envs ${TEST_TASK} --action-repeat 2 --id ${ALGO} --device cuda:${GPU_ID} --seed $seed --episodes ${EPISODES} --sep-replay-buffer True --reward-size ${REWARD_SIZE} --task-weight ${TASK_WEIGHT} --task-loss-type ${TASK_LOSS_TYPE} --context-dim ${CONTEXT_DIM}
done
echo "Experiments ended."

# run command
# ./generalization_run.sh tad walker-stand~walker-walk 0 2000 10.0 0.1 cross_entropy 0
# for ALGO, we can choose dreamer or planet or rad
# for TASK, we can choose
# for GPU_ID, we can choose 0~MAX_ID
# for EPISODES, we always choose 1000 or 2000
# for TASK_LOSS_TYPE, we can choose cross_entropy or supervised_contrastive

# tasks:
# 1.
# "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8
# ~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65"
# "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# 2.
# "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2
# ~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2"
# "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# 3.
# "mywalker-walk_speed-0.5-0.2~mywalker-walk_speed-1.5-0.2
# ~mywalker-walk_speed-2.0-0.2~mywalker-walk_speed-3.0-0.2"
# "mywalker-walk_speed-1.0-0.2~mywalker-walk_speed-2.5-0.2"
# 4.
# "myquadruped-walk_speed-0.5-0.2~myquadruped-walk_speed-1.5-0.2
# ~myquadruped-walk_speed-2.0-0.2~myquadruped-walk_speed-3.0-0.2"
# "myquadruped-walk_speed-1.0-0.2~myquadruped-walk_speed-2.5-0.2"


# E.g.
# ./generalization_run.sh planet
# "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8
# ~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65"
# "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# 0 2000 0.0 0.0

# ./generalization_run.sh dreamer
# "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8
# ~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65"
# "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# 1 2000 0.0 0.0

# ./generalization_run.sh rad
# "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8
# ~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65"
# "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# 2 2000 10.0 0.1

# ./generalization_run.sh rad
# "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8
# ~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65"
# "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# 3 2000 10.0 0.5

# ./generalization_run.sh planet
# "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2
# ~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2"
# "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# 4 2000 0.0 0.0

# ./generalization_run.sh dreamer
# "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2
# ~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2"
# "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# 5 2000 0.0 0.0

# ./generalization_run.sh rad
# "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2
# ~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2"
# "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# 6 2000 10.0 0.1

# ./generalization_run.sh rad
# "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2
# ~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2"
# "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# 7 2000 10.0 0.5

