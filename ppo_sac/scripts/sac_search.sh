#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <num_instances> <current_instance_index>"
  exit 1
fi

num_instances=$1
cur_instance=$2

# Register experiments: list of all logdirs and corresponding commands.
all_commands=()
for i in {1..3}; do
  for pool_size in 5 10 100 30000; do
    for aug in "crop" "shift" "channel_shuffle" "grayscale" "inversion" "color_jitter"; do
      # rad
      all_commands+=(
        "sac_rad_${aug}_w3_imagenet_p${pool_size}_${i} \
        --env-configs \"{\\\"image_pool_size\\\": $pool_size}\" \
        --apply-data-augmentation --augmentations $aug \
        --independent-encoder --encoder-tau 0.025"
      )
      # curl
      all_commands+=(
        "sac_curl_${aug}_w3_imagenet_p${pool_size}_${i} \
        --env-configs \"{\\\"image_pool_size\\\": $pool_size}\" \
        --apply-data-augmentation --augmentations $aug \
        --independent-encoder --encoder-tau 0.025 \
        --use-curl-loss --contrastive-positives augmented"
      )
      # spr
      all_commands+=(
        "sac_spr_${aug}_w3_imagenet_p${pool_size}_${i} \
        --env-configs \"{\\\"image_pool_size\\\": $pool_size}\" \
        --apply-data-augmentation --augmentations $aug \
        --independent-encoder --encoder-tau 0.025 \
        --use-spr-loss"
      )
    done
  done
done


# ---- SAC-SPECIFIC LOGIC

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
        echo "Skipping incomplete experiment folder $logdir"
        # rm -rf "$logdir"
      fi
    fi
    logdirs+=("$logdir")
    commands+=("python src/train_sac.py --run-name ${cmd}")
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