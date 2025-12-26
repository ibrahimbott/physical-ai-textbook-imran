---
title: "Module 3: Simulation & Digital Twins"
description: "Mastering the Art of Digital Robotics: Gazebo, Unity, and URDF"
module: 3
duration: "8-12 hours"
prerequisites: "ROS 2 basics"
objectives:
  - Compare and contrast Gazebo and Unity for robotic simulation.
  - Write a basic URDF to model physical links and joints.
  - Implement a digital twin with simulated sensors (Lidar, Camera).
  - Understand the 'Sim-to-Real' gap and strategies to overcome it.
---

# Module 3: Simulation & Digital Twins

Simulation is the bridge between design and reality. In this module you'll:training dojo for physical AI. Before a robot takes its first step in the real world, it walks a million miles in the matrix.

<img src="/img/module3_lidar.png" class="module-header-image" />

## 1. The Simulation Ecosystem

Why do we simulate?
1.  **Safety**: A falling humanoid breaks expensive parts. A falling simulated humanoid costs nothing.
2.  **Speed**: We can train agents faster than real-time.
3.  **Scalability**: We can train 1000 robots in parallel.

### Gazebo vs. Unity

| Feature | Gazebo (Classic/Ignition) | Unity (with Robotics Hub) |
| :--- | :--- | :--- |
| **Physics Engine** | ODE/Bullet (High precision for mechanics) | PhysX (Fast, stable, gaming-grade) |
| **Visual Fidelity** | Functional, basic rendering | Photorealistic, Ray-tracing suitable for Vision AI |
| **ROS Integration** | Native, seamless | Requires TCP/IP Bridge (ROS-TCP-Connector) |
| **Best For** | Testing dynamics, control loops, navigation | Human-Robot Interaction, Computer Vision datasets |

## 2. Modeling the Body: URDF

The **Unified Robot Description Format (URDF)** is the standard XML format for representing robot models in ROS.

### Anatomy of a URDF
Ref: [ROS 2 URDF Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)

A simple robot arm link:

```xml
<robot name="simple_arm">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint Definition -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>
```

> [!TIP]
> **Inertia Matrices Matter!** If your simulated robot wobbles or explodes, 90% of the time your `<inertial>` values are wrong. Use a solid modeling tool (like SolidWorks or Fusion 360) to calculate accurate moments of inertia.

## 3. Simulated Perception

To make a Digital Twin useful, it must see what the real robot sees.

### Extending URDF with Gazebo Plugins

To simulate a Lidar, we add a `<gazebo>` tag to our URDF:

```xml
<gazebo reference="lidar_link">
  <sensor type="ray" name="head_lidar">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>10.0</max>
      </range>
    </ray>
    <plugin name="gazebo_ros_head_lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## 4. The Sim-to-Real Gap

The "Sim-to-Real Gap" refers to the drop in performance when a policy trained in simulation is deployed on physical hardware.

**Common Causes:**
- **Friction Modeling**: Coulomb friction is hard to model perfectly.
- **Latency**: Simulation often ignores communication delays.
- **Sensor Noise**: Real cameras have blur, grain, and variable exposure; simulation is often perfect.

**Solution:** *Domain Randomization* (Covered in Module 6).
