#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <num_instances> <current_instance_index>"
  exit 1
fi

num_instances=$1
cur_instance=$2

# Register experiments: list of all logdirs and corresponding commands.
all_commands=()
for i in {1..5}; do
  # ONEHOT
  # w3
  all_commands+=("ppo_w3_onehot_${i} --env-configs \"{\\\"w\\\": 3, \\\"variation\\\": \\\"onehot\\\"}\" --hidden-layers 3")
  # w4
  all_commands+=("ppo_w4_onehot_${i} --env-configs \"{\\\"w\\\": 4, \\\"variation\\\": \\\"onehot\\\"}\" --hidden-layers 3")

  # w3
  for pool_size in 1 5 10 25 50 100; do
    all_commands+=("ppo_w3_imagenet_p${pool_size}_${i} --env-configs \"{\\\"w\\\": 3, \\\"image_pool_size\\\": $pool_size}\"")
  done
  for pool_size in 1 5 10; do
    all_commands+=("ppo_w3_diffusiondb_p${pool_size}_${i} --env-configs \"{\\\"w\\\": 3, \\\"image_pool_size\\\": $pool_size, \\\"image_folder\\\": \\\"diffusiondb\\\"}\"")
  done

  # w4
  for pool_size in 1 5 10; do
    all_commands+=("ppo_w4_imagenet_p${pool_size}_${i} --env-configs \"{\\\"w\\\": 4, \\\"image_pool_size\\\": $pool_size}\" --total-timesteps 50000000")
  done
done
all_commands+=("ppo_w3_imagenet_p3_1 --env-configs \"{\\\"w\\\": 3, \\\"image_pool_size\\\": 30000}\"")


# ---- PPO-SPECIFIC LOGIC

# Pre-filter experiments based on instance assignment.
logdirs=()
commands=()
for idx in "${!all_commands[@]}"; do
  cmd="${all_commands[$idx]}"
  # Extract logdir from the command string
  logdir="runs/$(echo "$cmd" | cut -d' ' -f1)"
  if (( idx % num_instances == cur_instance )); then
    if [ -d "$logdir" ]; then
      if [ -f "$logdir/done" ]; then
        echo "Skipping $logdir (already done)"
        continue
      else
        # echo "Deleting incomplete experiment folder $logdir"
        # rm -rf "$logdir"
        echo "Skipping $logdir (already exists)"
      fi
    fi
    logdirs+=("$logdir")
    commands+=("python src/train_ppo.py --run-name ${cmd}")
  fi
done


# ---- SCRIPT LOGIC


echo "About to run ${#logdirs[@]} experiments:"
for idx in "${!logdirs[@]}"; do
  echo "${logdirs[$idx]}: ${commands[$idx]}"
done

# Iterate over the filtered experiments.
for idx in "${!logdirs[@]}"; do
  logdir="${logdirs[$idx]}"
  cmd="${commands[$idx]}"
  echo "Running experiment in $logdir"
  echo "Command: $cmd"
  mkdir -p "$logdir"
  # Launch and log output
  eval "$cmd 2>&1 | tee \"$logdir/run.log\""
done