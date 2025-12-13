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
  all_commands+=("./logdir/dreamer_w3_onehot_${i} --configs spgym")
  # w4
  all_commands+=("./logdir/dreamer_w4_onehot_${i} --configs spgym --env.spgym.w 4")

  # IMAGENET
  # w3
  for pool_size in 1 5 10 20 30 50 100; do
    all_commands+=("./logdir/dreamer_w3_imagenet_p${pool_size}_${i} --configs spgym_image --env.spgym.image_pool_size ${pool_size}")
  done
  # no decoder
  for pool_size in 1 5 10; do # 25 50 100
    all_commands+=("./logdir/dreamer_w3_imagenet_p${pool_size}_${i} --configs spgym_image_nodec --env.spgym.image_pool_size ${pool_size}")
  done
  # w4
  for pool_size in 1 5 10; do
    all_commands+=("./logdir/dreamer_w4_imagenet_p${pool_size}_${i} --configs spgym_image --env.spgym.image_pool_size ${pool_size} --env.spgym.w 4")
  done
done
for pool_size in 10000 20000 30000; do
  all_commands+=("./logdir/dreamer_w3_imagenet_p${pool_size}_1 --configs spgym_image --env.spgym.image_pool_size ${pool_size}")
done

# ---- DREAMER-SPECIFIC LOGIC


# Pre-filter experiments based on instance assignment.
logdirs=()
commands=()
for idx in "${!all_commands[@]}"; do
  cmd="${all_commands[$idx]}"
  # Extract logdir from the command string by finding the token after "--logdir"
  logdir=$(echo "$cmd" | cut -d' ' -f1)
  if (( idx % num_instances == cur_instance )); then
    # Skip if the experiment finished
    if [ -f "$logdir/done" ]; then
        echo "Skipping $logdir (already done)"
        find "$logdir" -type f -name "*.npz" -delete
        continue
    fi
    if [ -d "$logdir" ]; then
        echo "Skipping $logdir (already exists)"
        continue
    fi
    logdirs+=("$logdir")
    commands+=("python dreamerv3/main.py --logdir ${cmd}")
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