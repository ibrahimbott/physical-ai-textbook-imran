---
title: "Resources"
description: "Hardware Bill of Materials and Software Tools"
---

# Resources

## Hardware Bill of Materials (BOM)

To build the "Physical AI" stack, here are recommended components at different budget levels.

### The Brain (Compute)
| Component | Budget/Starter | Pro/Research | Notes |
| :--- | :--- | :--- | :--- |
| **Main Computer** | **Jetson Orin Nano (8GB)** | **Jetson AGX Orin (64GB)** | Nano can run basic VLA (quantized); AGX runs full 7B+ models. |
| **Microcontroller** | ESP32 / Arduino Mega | Teensy 4.1 | For low-level motor control loops (1kHz+). |

### The Eyes (Sensors)
| Component | Recommendation | Why? |
| :--- | :--- | :--- |
| **RGB-D Camera** | **Intel RealSense D435i** | Reliability, standard ROS drivers, global shutter (good for motion). |
| **Lidar (2D)** | **RPLIDAR A1/A2** | Cheap, effective for 2D SLAM navigation. |
| **IMU** | BNO055 / Built-in to Camera | Essential for balancing humanoids. |

### The Body (Actuators)
*   **Dynamixel XL430-W250**: Great smart servo for small arms/humanoids. Supports Tuple Control.
*   **Feetech STS3215**: Budget alternative to Dynamixel (~$20 vs $50), reliable serial bus.
*   **ODrive + BLDC**: For high-power quadruped legs (requires custom mechanical design).

## Software Tools
*   **[Foxglove Studio](https://foxglove.dev/)**: Modern alternative to RViz for visualizing ROS data.
*   **[PlotJuggler](https://github.com/facontidavide/PlotJuggler)**: Essential for graphing data (PID tuning) in real-time.
*   **[Docker](https://www.docker.com/)**: Highly recommended to keep your ROS environment clean.
