---
title: "Module 6: Advanced AI & Control"
description: "Reinforcement Learning, PPO, and Sim-to-Real Transfer"
module: 6
duration: "8-12 hours"
prerequisites: "Module 3 (Sim), Module 5 (VLA)"
objectives:
  - Formulate a robotic task as a Markov Decision Process (MDP).
  - Train a locomotion policy using Proximal Policy Optimization (PPO).
  - Implement Domain Randomization to make policies robust.
  - Design a hierarchical control stack (Planner -> Policy -> PD).
---

# Module 6: Advanced AI & Control

While VLAs give high-level reasoning ("Pick up the cup"), **Reinforcement Learning (RL)** gives the athletic intelligence ("Balance on one foot while reaching").

<img src="/img/module6_rl.png" class="module-header-image" />

## 1. Reinforcement Learning Basics

We frame robot control as an MDP:
- **State ($S_t$)**: Joint angles, velocities, IMU orientation.
- **Action ($A_t$)**: Joint position targets or torques.
- **Reward ($R_t$)**: $+1$ for moving forward, $-10$ for falling.

### The PPO Algorithm
**Proximal Policy Optimization** is the workhorse of modern robotics. It is stable and reasonably sample-efficient.
- It prevents the policy from changing *too much* in dynamic steps, stopping the robot from learning catastrophic behaviors (like jittering uncontrollably).

## 2. Domain Randomization (DR)

A policy trained in a perfect simulation will fail in reality. The real world has rough floors, heavy cables, and non-ideal motors.

**DR Strategy**: Train the agent in a "Multiverse" of simulations.
- **Mass**: Vary link masses by $\pm 20\%$.
- **Friction**: Vary floor friction from ice (0.1) to rubber (1.0).
- **Latency**: Inject random delays (10ms - 50ms) in the observation loop.
- **Perturbations**: Apply random pushes to the robot's torso during training.

*If the agent can walk on ice, with a heavy backpack, while being pushed... it can walk on the office floor.*

## 3. Hierarchical Control Architectures

We rarely use one neural network for everything. A typical stack:

| Layer | Component | Frequency | Responsibility |
| :--- | :--- | :--- | :--- |
| **High** | VLA / LLM | 0.5 - 1 Hz | "Go to the kitchen" (Reasoning) |
| **Mid** | RL Policy | 50 Hz | Walking gait, obstacle avoidance (Athletics) |
| **Low** | PID / Motor Driver | 1000 Hz | Tracking joint position targets (Reflexes) |

This separation concerns ensures that if the LLM "thinks" too slowly, the robot doesn't fall over because the spinal cord (RL/PID) is still running fast.
