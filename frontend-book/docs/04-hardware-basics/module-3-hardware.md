---
title: "Module 4: Hardware Foundations"
description: "The silicon brain and steel muscles: Jetson, Sensors, and Actuators"
module: 4
duration: "6-8 hours"
prerequisites: "Basic Electronics, Linux"
objectives:
  - Understand the architecture of Edge AI computers (NVIDIA Jetson).
  - Select appropriate sensors (Depth Cameras, IMUs, Lidar) for humanoid tasks.
  - Differentiate between actuation technologies (BLDC, Stepper, Servo).
  - Plan power distribution for a mobile robot.
---

# Module 4: Hardware Foundations

Software defines what a robot *can* do; hardware defines what it *looks* like and how long it survives. In this module, we dissect the physical components of a humanoid.

<img src="/img/hardware_chip.png" class="module-header-image" />

## 1. The Compute Core: NVIDIA Jetson

For modern Physical AI, the CPU is not enough. We need parallel processing for Vision Transformers and RL inference.

### Why Jetson?
Platforms like the **Jetson Orin Nano** or **AGX Orin** provide:
- **Unified Memory Architecture**: CPU and GPU share the same RAM. Zero-copy transfer means no latency moving images from camera buffer to GPU memory.
- **Tensor Cores**: Specialized hardware for matrix multiplication, accelerating PyTorch/TensorFlow operations by 10x over standard CUDA cores.

#### Comparative Specs
| Board | AI Performance (TOPS) | GPU Architecture | RAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Raspberry Pi 5** | ~0.5 (CPU) | VideoCore (Not CUDA) | 4-8 GB | Low-level Control, ROS bridging |
| **Jetson Orin Nano** | 40 | Ampere (1024 cores) | 8 GB | Entry-level Vision, VSLAM |
| **Jetson AGX Orin** | 275 | Ampere (2048 cores) | 32-64 GB | LLM Inference (7B models), Full Humanoid Control |

## 2. Sensors: The Eyes and Ears

### Depth Cameras (RGB-D)
We rely on depth to build 3D maps (Octomaps).
- **Stereo Depth (e.g., RealSense D435i)**: Uses two lenses to calculate disparity. Works well outdoors.
- **Structured Light (e.g., RealSense D455)**: Projects an IR pattern. Extremely precise indoors, fails in direct sunlight.
- **Time of Flight (ToF)**: Measures light pulse return time. fast, but noisy at edges.

> [!IMPORTANT]
> **Bandwidth Warning**: A single 1080p RGB-D stream can consume 300+ MB/s. Ensure your USB cables are rated for **USB 3.1 Gen 1 or Gen 2**. Ordinary charging cables will cause the camera to drop frames or fail to enumerate.

## 3. Actuation: Making it Move

### Smart Servos (Dynamixel / Boleo)
Unlike hobby servos (PWM), smart servos use serial communication (TTL/RS-485).
- **Feedback**: They report Position, Velocity, Temperature, and Load.
- **Control Modes**: We can command Goal Position (Position Control) or Goal Current (Torque Control).

**Torque Control** is critical for safe Human-Robot Interaction. It allows the robot to be "compliant"—if you push it, it yields rather than fighting you.

## 4. Power Management

A humanoid is an energy-hungry beast.
- **Logic Power**: 5V/3A stable for Jetson + Sensors. Use a dedicated BEC (Battery Eliminator Circuit).
- **Motor Power**: 12V-24V high current (often 20A-60A bursts).
- **battery Safety**: LiPo batteries can explode if over-discharged. Always actuate via a BMS (Battery Management System).
