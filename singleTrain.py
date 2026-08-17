# Train a Soft Actor-Critic (SAC) agent to navigate an environment (AntMaze_Medium-v5).
# Uses Hindsight Experience Replay (HER) and curriculum learning to
# progressively increase goal difficulty during training.

import gymnasium as gym
import os
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from goalCurriculum import AntMazeCurriculumWrapper, CurriculumCallback

# Create AntMaze environment with curriculum learning
def make_env():
    env = gym.make("AntMaze_Medium-v5")

    # Apply curriculum learning wrapper to progressively increase goal difficulty
    env = AntMazeCurriculumWrapper(env)
    return env

# Vectorized environment required by Stable-Baselines3
env = DummyVecEnv([make_env])

# Configure model and TensorBoard logging directories
models_dir = "antModel/SAC"
logdir = "logs1"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize SAC with Hindsight Experience Replay (HER)
model = SAC(
    "MultiInputPolicy", 
    env=env, 
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),

    learning_rate=3e-4,
    batch_size=256,
    learning_starts=5_000,
    buffer_size=1_000_000,
    ent_coef="auto_0.1",
    tau=0.005,
    gamma=0.99,
    tensorboard_log=logdir,
    verbose=1,
)

# Parameters for total timesteps and checkpoint locations
timesteps = 5_000_000
checkpoints = [timesteps//4 * i for i in range(1, 5)]

curriculum_callback = CurriculumCallback(
    env,
    models_dir=models_dir,
    total_timesteps = timesteps,
    check_freq = 5000,
    success_threshold = .6,
    )

model.learn(total_timesteps=timesteps, 
            callback=curriculum_callback,
            #reset_num_timesteps=False # when I want to continue training on a model
            )

# Save final trained policy
model.save(f"{models_dir}/antMedium{timesteps}")