# Evaluate the success rate of a 4-legged robot (8-dimensional action space)
# and a point (2-dimensional action space) over 20 fixed seeds.
# Both agents are trained using Soft Actor-Critic (SAC),
# Hindsight Experience Replay (HER), and curriculum learning.

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def make_env(env_name):
    env = gym.make(env_name, render_mode=None)
    return env

def run_simulation(env, model, max_episodes=20, seedNum=0):
    # Start a new episode (random start + goal states)
    seedNum = 0
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

################# Ant Maze #############
print("AntMaze 20 Episode Evaluation")
ENV_NAME = "AntMaze_Medium-v5"
MODEL_NAME = "models/SAC/antMazeMed_SAC_10M"

env = DummyVecEnv([lambda: make_env(ENV_NAME)])
model = SAC.load(MODEL_NAME, env=env)
run_simulation(env, model, 20)

################# Point Maze ###########
print("\n\nPointMaze 20 Episode Evaluation")
ENV_NAME = "PointMaze_Medium-v3"
MODEL_NAME = "pointModel/SAC/pointLarge100000"

env = DummyVecEnv([lambda: make_env(ENV_NAME)])
model = SAC.load(MODEL_NAME, env=env)
run_simulation(env, model, 20)
