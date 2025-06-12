import glob
import random
import os

import gymnasium as gym
import numpy as np
import pickle
import torch
from tqdm import tqdm
import yaml

import sliding_puzzles
import augmentations as a
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
    if not "images" in args.env_configs or args.env_configs["images"] is None or len(args.env_configs["images"]) == 0:
        raise ValueError("No images found in env_configs")
    envs.set_attr("images", [args.env_configs["images"] for _ in range(envs.num_envs)])

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
        raise TypeError(f"Unknown algorithm: {run_name}")

env_make_list = []
for i in range(num_eval_envs):
    # each environment has its unique image
    ood_env_configs = {"seed": random.randint(0, 2147483647), "image_pool_seed": 1, "variation": "image", "image_folder": "imagenet-1k", "image_size": 100, "w": 3, "image_pool_size": 1}
    env_make_list.append(e.make("SlidingPuzzles-v0", i, False, "", ood_env_configs))
envs = gym.vector.AsyncVectorEnv(env_make_list)


def evaluate(agent, envs, augs=[]):
    device = next(agent.parameters()).device
    obs, _ = envs.reset()
    episodes = 0
    returns = 0.0
    lengths = 0.0
    successes = 0.0
    pbar = tqdm(range(envs.get_attr("max_steps")[0]), desc="Eval steps")
    while episodes < envs.num_envs:
        pbar.update(1)

        obs = torch.Tensor(obs).to(device) / 255.0

        for aug in augs:
            obs = getattr(a, aug)(obs)

        if obs.shape[-1] != 84:
            obs = torch.nn.functional.interpolate(obs, size=(84, 84), mode='bilinear', align_corners=False)

        with torch.no_grad():
            agent_actions = agent(obs)[0].detach().cpu().numpy()

        obs, reward, terminated, truncated, info = envs.step(agent_actions)

        if "episode" in info:
            for i, done in enumerate(info["_episode"]):
                if done:
                    returns += info["episode"]["r"][i]
                    lengths += info["episode"]["l"][i]
                    episodes += 1
                    pos_str = f"r={(returns/episodes):.2f}"
                    if "is_success" in info:
                        pos_str += f", s={(successes/episodes):.2f}"
                        successes += info["is_success"][i]
                    pbar.set_postfix_str(pos_str)
    return returns, lengths, successes, episodes


import itertools

# Define the augmentations
augmentations = ["crop", "shift", "channel_shuffle", "grayscale", "inversion"]
max_augmentations = 2

# Generate all combinations
all_combinations = [()]
for r in range(1, min(max_augmentations, len(augmentations)) + 1):
    # Get combinations of size r
    combos = list(itertools.combinations(augmentations, r))
    all_combinations.extend(combos)
    
    # Add reverse order for each combination
    for combo in combos:
        reversed_combo = tuple(reversed(combo))
        if reversed_combo != combo and reversed_combo not in all_combinations:
            all_combinations.append(reversed_combo)

result_file = "ood_eval_easy_results.pkl"
if os.path.exists(result_file):
    with open(result_file, "rb") as f:
        results = pickle.load(f)
else:
    results = {}

runs_to_process = [f for f in os.listdir("runs") if os.path.isdir(f"runs/{f}")]

for run_path in runs_to_process:
    if run_path not in results:
        results[run_path] = {}

    actor = None
    for combo in all_combinations:
        combo_str = ",".join([str(x) for x in combo])
        if combo_str in results[run_path]:
            print(f"{run_path}: {combo_str} already exists")
            continue

        print("\t", combo_str)
        if actor is None:
            try:
                actor = load_run(run_path, envs)
            except ValueError as e:
                print(f"{run_path}: {e}")
                break

        returns, lengths, successes, episodes = evaluate(actor, envs, combo)
        results[run_path][combo_str] = {
            "successes": successes,
            "episodes": episodes,
            "returns": returns,
            "lengths": lengths
        }

        # Save results to pickle file
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)