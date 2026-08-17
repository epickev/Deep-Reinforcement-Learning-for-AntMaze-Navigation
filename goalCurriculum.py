import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

class AntMazeCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, initial_max_dist=5.0, dist_step=1.0):
        super().__init__(env) # points self.env to the raw antMaze 
        self.current_max_dist = initial_max_dist
        self.dist_step = dist_step
        self.last_successes = deque(maxlen=50)
        self.episode_count = 0
    
    # Override the core env.reset() function, not the vectored env
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        currPos = obs['achieved_goal']
        goalPos = obs['desired_goal']
        dist = np.linalg.norm(goalPos - currPos)
        self.episode_count = self.episode_count + 1

        # Keep resetting environment until distance < max distance
        while dist > self.current_max_dist:
            obs, info = self.env.reset(**kwargs)
            currPos = obs['achieved_goal']
            goalPos = obs['desired_goal']
            dist = np.linalg.norm(goalPos - currPos)

        print(f"Episode {self.episode_count}, Dist: {dist}, Max D: {self.current_max_dist}")

        return obs, info
    
    # Get success rate and append to deque
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # At the end of an episode, check if the agent won
        if terminated or truncated:
            success = info.get("success", False)
            self.last_successes.append(float(success))
            
        return obs, reward, terminated, truncated, info

    def set_max_distance(self, dist):
        self.current_max_dist = dist

class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, models_dir, total_timesteps, check_freq, success_threshold, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.models_dir = models_dir
        self.success_threshold = success_threshold

        # Save every quarter
        self.save_freq = total_timesteps//4
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            # Get Success rate from wrapper deque
            current_success_rate = np.mean(self.training_env.get_attr("last_successes"))

            # Print deque + Success rate at each freq checks
            print(self.training_env.get_attr("last_successes"))
            print("Success rate: ", current_success_rate)
            
            if current_success_rate > self.success_threshold:
                old_dist = self.training_env.get_attr("current_max_dist")[0]
               
                # Increment the max distance by 1
                self.training_env.set_attr("current_max_dist", old_dist + 1.0)
                print(f"Curriculum updated: Max goal distance is now {old_dist + 1.0}")
            
            # Save model every quarter of training
            checkpoints = [ self.total_timesteps//4 * i for i in range(1, 5)]
            if self.n_calls in checkpoints:
                print(f"Reached timestep: {self.n_calls}, Model Saved")
                self.model.save(f"{self.models_dir}/antSmall_{self.num_timesteps}Success_{current_success_rate}")
                
        return True