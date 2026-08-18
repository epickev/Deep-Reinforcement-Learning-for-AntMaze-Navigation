# Deep-Reinforcement-Learning-for-AntMaze-Navigation

## Demo
[![Deep RL AntMaze Navigation Demo](https://img.youtube.com/vi/NfCCaBIAsm0/hqdefault.jpg)](https://www.youtube.com/watch?v=NfCCaBIAsm0)

## Overview

Developed goal-conditioned reinforcement learning agents for maze navigation using Soft Actor-Critic (SAC), Hindsight Experience Replay (HER), and curriculum learning. Evaluated navigation performance across AntMaze and PointMaze environments using 20 fixed evaluation seeds.

**Architecture:**  
Maze Environment → Curriculum Learning → SAC + HER → Policy Training → Fixed-Seed Evaluation → Success Rate

## Key Features & Results
- Trained a quadrupedal Ant agent with an 8-dimensional continuous action space and a Point agent with a 2-dimensional continuous action space
- Combined Soft Actor-Critic (SAC) with Hindsight Experience Replay (HER) for goal-conditioned navigation
- Applied curriculum learning to progressively increase task difficulty during training
- Evaluated policy performance using 20 fixed random seeds for consistent comparison
- Measured navigation performance using episode success rate at intermediate and final training checkpoints
- Compared learning progression between AntMaze and PointMaze agents

**Evaluation Results:**

| Environment | Training Steps | Success Rate |
|---|---:|---:|
| AntMaze | 5M | 0% |
| AntMaze | 10M | 5% |
| PointMaze | 40K | 60% |
| PointMaze | 100K | 70% |

**Result:** PointMaze achieved 70% success after 100K training steps, while AntMaze achieved 5% after 10M steps, highlighting the substantially greater difficulty of learning locomotion and navigation with an 8-dimensional action space.
