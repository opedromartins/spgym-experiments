import glob
import random
import os

import gymnasium as gym
import numpy as np
import pickle
import torch
import yaml

import sliding_puzzles
import utils as u
import environments as e
from algorithms import SACDiscrete
from train_ppo import Agent as PPO

num_eval_envs = 100
device = "cuda"

class Args:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

def load_run(run_name, envs):
    args = Args(yaml.load(open(run_name + "/config.yaml", "r"), Loader=yaml.FullLoader))
    last_checkpoint = max([int(x.split("_")[1].split(".")[0]) for x in os.listdir(run_name) if x.startswith("checkpoint_")])
    path = f"{run_name}/checkpoint_{last_checkpoint}.pth"
    print("Loading checkpoint from", path)
    checkpoint = torch.load(path, map_location=device)
    if "sac" in run_name.lower():
        agent = SACDiscrete(envs, args, None, device, print_nets=False)
        agent.load_state_dict(checkpoint["agent"])
        return agent.actor
    elif "ppo" in run_name.lower():
        agent = PPO(envs, args.env_id, args.hidden_size, args.hidden_layers, args.min_res)
        agent.load_state_dict(checkpoint["agent"])
        return agent
    else:
        raise ValueError(f"Unknown algorithm: {run_name}")
    
env_make_list = []
for i in range(num_eval_envs):
    # each environment has its unique image
    ood_seed = random.randint(0, 2147483647)
    ood_env_configs = {"seed": ood_seed, "image_pool_seed": ood_seed, "variation": "image", "image_folder": "imagenet-1k", "image_size": 84, "w": 3, "image_pool_size": 1}
    env_make_list.append(e.make("SlidingPuzzles-v0", i, False, "", ood_env_configs))
envs = gym.vector.SyncVectorEnv(env_make_list)

runs_to_process = [f for f in os.listdir("runs") if os.path.isdir(f"runs/{f}")]

results = {}
for run_path in runs_to_process:
    results[run_path] = {}
    actor = load_run(run_path, envs)

    returns, lengths, successes, episodes = u.evaluate(actor, envs)
    results[run_path] = {
        "successes": successes,
        "episodes": episodes,
        "returns": returns,
        "lengths": lengths
    }

    # Save results to pickle file
    with open('ood_eval_hard_results.pkl', 'wb') as f:
        pickle.dump(results, f)