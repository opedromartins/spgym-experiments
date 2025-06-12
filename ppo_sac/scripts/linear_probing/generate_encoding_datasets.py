import base64
import glob
import random
import os
import resource
from concurrent.futures import ProcessPoolExecutor
import functools
import multiprocessing
import gc
import json
import gymnasium as gym
import gzip
import numpy as np
import pickle
import torch
from tqdm import tqdm
import yaml
import zlib

import sliding_puzzles
import utils as u
import environments as e
from algorithms import SACDiscrete
from train_ppo import Agent as PPO

device = "cuda"
samples = 50000
num_eval_envs = 1000

class Args:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
print(f"Soft file descriptor limit: {soft}, Hard limit: {hard}")

# Create environments
env_make_list = []
for i in range(num_eval_envs):
    env_configs = {"seed": random.randint(0, 2147483647), "image_pool_seed": 1, "variation": "image", "image_folder": "imagenet-1k", "image_size": 84, "w": 3, "image_pool_size": 1}
    env_make_list.append(e.make("SlidingPuzzles-v0", i, False, "", env_configs))
envs = gym.vector.AsyncVectorEnv(env_make_list)

# Collect all runs to process
runs_to_process = [f for f in os.listdir("runs") if os.path.isdir(f"runs/{f}")]

# Process each run sequentially
os.makedirs("encoding_datasets", exist_ok=True)
for run_name in tqdm(runs_to_process, desc="Runs"):
    if os.path.exists(f"encoding_datasets/{run_name}.jsonl"):
        # print(f"Skipping {run_name} because it already exists")
        continue

    # Find the run folder
    for folder in glob.glob("runs*/"):
        run_path = os.path.join(folder, run_name)
        try:
            args = Args(yaml.load(open(run_path + "/config.yaml", "r"), Loader=yaml.FullLoader))
            break
        except FileNotFoundError:
            continue
    else:
        print(f"Could not find config.yaml for {run_name}")
        continue

    print(f"Processing {run_path}")

    # Load config and set env images
    if not "images" in args.env_configs or args.env_configs["images"] is None or len(args.env_configs["images"]) == 0:
        print("No images found in env_configs")
        continue

    envs.set_attr("images", [args.env_configs["images"] for _ in range(envs.num_envs)])
    assert envs.get_attr("images")[0] == args.env_configs["images"]

    # Load checkpoint
    last_checkpoint = max([int(x.split("_")[1].split(".")[0]) for x in os.listdir(run_path) if x.startswith("checkpoint_")])
    path = f"{run_path}/checkpoint_{last_checkpoint}.pth"
    print("Loading checkpoint from", path)
    checkpoint = torch.load(path, map_location=device)
    
    if "sac" in run_name.lower():
        agent = SACDiscrete(envs, args, None, device, print_nets=False)
        agent.load_state_dict(checkpoint["agent"])
        actor = agent.actor
    elif "ppo" in run_name.lower():
        actor = PPO(envs, args.env_id, args.hidden_size, args.hidden_layers, args.min_res)
        actor.load_state_dict(checkpoint["agent"])
    else:
        raise ValueError(f"Unknown algorithm: {run_name}")

    # Extract latents and save as we go
    actor.encoder.to(device)
    filename = f"encoding_datasets/{run_name}.jsonl"
    collected = 0
    pbar = tqdm(range(samples), desc="Eval steps")
    unique_combinations = set()
    
    with open(filename, 'w') as f:
        while collected < samples:
            obs, info = envs.reset()

            obs = torch.Tensor(obs).to(device) / 255.0
            if obs.shape[-1] != 84:
                obs = torch.nn.functional.interpolate(obs, size=(84, 84), mode='bilinear', align_corners=False)

            with torch.no_grad():
                latent = actor.encoder(obs).detach().cpu().numpy()

            # Process and save each sample individually
            for i in range(len(obs)):
                # Convert obs to uint8 and compress using zlib
                obs_array = (obs[i].cpu().numpy() * 255).astype(np.uint8)
                compressed_obs = zlib.compress(obs_array.tobytes())
                # Encode compressed bytes as base64 for JSON compatibility
                encoded_obs = base64.b64encode(compressed_obs).decode('ascii')
                state = info["state"][i].tolist()

                # Create unique combination key
                state_str = json.dumps(state)
                combination = (encoded_obs, state_str)
                
                # Only save if combination is unique
                if combination not in unique_combinations:
                    unique_combinations.add(combination)
                    sample = {
                        "obs": encoded_obs,
                        "latent": latent[i].tolist(),
                        "state": state
                    }
                    f.write(json.dumps(sample) + '\n')
                    collected += 1
                    pbar.update(1)
                    if collected >= samples:
                        break

    print(f"Saved dataset to {filename}")

    actor.cpu()
    del actor
    gc.collect()