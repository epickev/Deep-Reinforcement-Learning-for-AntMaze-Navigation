# Compare navigation performance at intermediate and final training checkpoints
# for AntMaze and PointMaze agents using fixed evaluation seeds.
# Visualizes the progression of learned policies from partial to final training.

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time

def make_env(env_name):
    env = gym.make(env_name, render_mode="human")
    return env

def run_simulation(env, model, max_episodes=20, seedNum=0):
    # Start a new episode (random start + goal states)
    env.seed(seedNum)
    obs = env.reset()
    episode_count = 0
    successes = []

    while episode_count < max_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if info[0]['success']:
            print(f"Seed:{seedNum}, Goal Reached!!!!!!!!!")
            success = 1
            episode_count += 1
            seedNum += 1
            successes.append(success)
            env.seed(seedNum)
            obs = env.reset()
        
        if done:
            print(f"Max time Reached: ")
            success = 0
            episode_count += 1
            seedNum += 1
            successes.append(success)
            env.seed(seedNum)
            obs = env.reset()

    env.close()
    print(successes)
    print("success rate: ", np.mean(successes))

########## Ant Maze 50% training ###########
ENV_NAME = "AntMaze_Medium-v5"
MODEL_NAME = "models/SAC/sac_antmaze_5000000_steps"
SEED = 1
NUM_ITER = 1

env = DummyVecEnv([lambda: make_env(ENV_NAME)])
model = SAC.load(MODEL_NAME, env=env)
run_simulation(env, model, NUM_ITER, SEED)
time.sleep(1)

########## Ant Maze Final  ###########
ENV_NAME = "AntMaze_Medium-v5"
MODEL_NAME = "models/SAC/antMazeMed_SAC_10M"

env = DummyVecEnv([lambda: make_env(ENV_NAME)])
model = SAC.load(MODEL_NAME, env=env)
run_simulation(env, model, NUM_ITER, SEED)


######### Point Maze 40% Training #########
ENV_NAME = "PointMaze_Medium-v3"
MODEL_NAME = "pointModel/SAC/pointLarge40000"
SEED = 0
NUM_ITER = 1

env = DummyVecEnv([lambda: make_env(ENV_NAME)])
model = SAC.load(MODEL_NAME, env=env)
run_simulation(env, model, NUM_ITER, SEED)
time.sleep(1)

######### Point Maze Final #############
MODEL_NAME = "pointModel/SAC/pointLarge100000"

env = DummyVecEnv([lambda: make_env(ENV_NAME)])
model = SAC.load(MODEL_NAME, env=env)
run_simulation(env, model, NUM_ITER, SEED)


