---
sidebar_label: 'Humanoid Structure with URDF'
sidebar_position: 4
---

# Humanoid Structure with URDF

## Learning Objectives

- Understand URDF (Unified Robot Description Format) and its purpose
- Identify and explain the core elements: links, joints, and transmissions
- Interpret humanoid robot structure from URDF files
- Recognize materials, visual properties, and collision properties in URDF
- Understand the connection between software representation and physical robot body

## Introduction to URDF

**URDF (Unified Robot Description Format)** is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including its structure, joints, inertial properties, and visual appearance. URDF serves as the bridge between the abstract software model of a robot and its physical reality.

### Purpose of URDF

- **Robot modeling**: Define the kinematic and dynamic properties of robots
- **Simulation**: Provide robot descriptions for physics simulators like Gazebo
- **Visualization**: Enable 3D visualization of robots in tools like RViz
- **Kinematic computation**: Support forward and inverse kinematics calculations
- **Collision detection**: Define collision properties for safety and planning

## Core URDF Elements

### Links

**Links** represent rigid components of a robot body. They are the building blocks that form the robot's structure.

#### Link Properties

- **Inertial properties**: Mass, center of mass, and inertia tensor
- **Visual properties**: How the link appears in visualization
- **Collision properties**: How the link interacts in collision detection

#### Example Link Definition

```xml
<link name="base_link">
  <inertial>
    <mass value="1.0" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01" />
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <cylinder length="0.1" radius="0.2" />
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1" />
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <cylinder length="0.1" radius="0.2" />
    </geometry>
  </collision>
</link>
```

### Joints

**Joints** define the connections between links and specify how they can move relative to each other. They establish the kinematic relationships in the robot.

#### Joint Types

- **Revolute**: Rotational joint with limited range of motion
- **Continuous**: Rotational joint with unlimited range of motion
- **Prismatic**: Linear sliding joint with limited range of motion
- **Fixed**: No movement between links (rigid connection)
- **Floating**: 6-DOF connection (rarely used)
- **Planar**: Movement on a plane (rarely used)

#### Joint Properties

- **Parent and child links**: Defines the connection between two links
- **Origin**: Position and orientation of the joint relative to the parent
- **Axis**: Direction of motion for the joint
- **Limits**: Range of motion and maximum effort/velocity for moveable joints

#### Example Joint Definition

```xml
<joint name="base_to_wheel" type="continuous">
  <parent link="base_link" />
  <child link="wheel_link" />
  <origin xyz="0.1 0 0" rpy="0 0 0" />
  <axis xyz="0 1 0" />
</joint>
```

### Transmissions

**Transmissions** define the mapping between actuators (motors) and joints. They describe how power is transferred from actuators to joints.

#### Transmission Types

- **Simple transmission**: Direct mapping between actuator and joint
- **Differential transmission**: Mapping for differential drive systems
- **Four-bar linkage transmission**: For complex mechanical linkages

#### Example Transmission Definition

```xml
<transmission name="wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Materials, Visual Properties, and Collision Properties

### Materials

Materials define the visual appearance of robot components:

```xml
<material name="red">
  <color rgba="1 0 0 1" />
</material>
<material name="blue">
  <color rgba="0 0 1 1" />
</material>
```

### Visual Properties

Visual elements define how the robot appears in visualization tools:

- **Geometry**: Shape (box, cylinder, sphere, mesh)
- **Origin**: Position and orientation relative to the link
- **Material**: Color and appearance properties

### Collision Properties

Collision elements define how the robot interacts with the environment in simulation:

- **Geometry**: Shape used for collision detection
- **Origin**: Position and orientation relative to the link
- **Surface properties**: Friction, restitution, etc.

## Humanoid-Specific Considerations

### Common Joint Types for Humanoid Robots

Humanoid robots have specific joint configurations that mimic human anatomy:

- **6-DOF joints**: For complex movements in shoulders and hips
- **3-DOF joints**: For simplified arm and leg movements
- **Revolute joints**: For single-axis rotation (elbows, knees)
- **Fixed joints**: For rigid connections (hands to forearms)

### Kinematic Chains for Arms and Legs

Humanoid robots typically have kinematic chains that mirror human limbs:

#### Arm Chain
```
torso -> shoulder -> upper_arm -> elbow -> forearm -> wrist -> hand
```

#### Leg Chain
```
torso -> hip -> thigh -> knee -> shin -> ankle -> foot
```

### Sensor Integration in URDF

Sensors are represented as special links in URDF:

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.08 0.04" />
    </geometry>
  </visual>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head_link" />
  <child link="camera_link" />
  <origin xyz="0.05 0 0.1" rpy="0 0 0" />
</joint>
```

### Gazebo Simulation Compatibility

URDF files often include Gazebo-specific extensions:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <self_collide>true</self_collide>
</gazebo>
```

## Interpreting Humanoid URDF Files

### Reading a URDF File Structure

When interpreting a humanoid URDF file, look for:

1. **Base link**: The root of the kinematic tree
2. **Kinematic chains**: Sequences of links and joints forming limbs
3. **Joint limits**: Range of motion constraints
4. **Mass properties**: Inertial parameters for simulation
5. **Sensor placements**: Locations of cameras, IMUs, etc.

### Example Humanoid Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link" />

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link" />
    <child link="torso" />
    <origin xyz="0 0 0.5" />
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6" />
      </geometry>
    </visual>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso" />
    <child link="head" />
    <origin xyz="0 0 0.35" />
    <axis xyz="0 1 0" />
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1" />
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1" />
      </geometry>
    </visual>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso" />
    <child link="left_upper_arm" />
    <origin xyz="0.2 0 0.2" />
    <axis xyz="1 0 0" />
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1" />
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05" />
      </geometry>
    </visual>
  </link>

  <!-- Additional joints and links for complete humanoid structure -->
</robot>
```

## Connection Between Software Representation and Physical Robot Body

### Software-to-Physical Mapping

URDF serves as the critical link between abstract software models and physical robots:

- **Kinematic model**: Software can calculate joint positions and end-effector poses
- **Dynamic model**: Simulation can predict robot behavior based on physical properties
- **Collision avoidance**: Planning algorithms use URDF to avoid self-collisions
- **Visualization**: Tools can render the robot in 3D space

### Calibration and Validation

The URDF model should match the physical robot:

- **Physical measurements**: Link lengths, joint positions, and masses
- **Joint calibration**: Ensuring software and physical joint angles align
- **Inertial properties**: Validating mass and center of mass for accurate simulation

## Best Practices for URDF Development

### File Organization

- Use Xacro for complex robots to avoid repetition
- Separate files for different robot parts
- Include proper documentation and comments

### Modeling Considerations

- Keep visual and collision models simple for performance
- Use accurate inertial properties for simulation
- Ensure proper joint limits match physical constraints
- Validate the kinematic tree has no loops

### Testing and Validation

- Use `check_urdf` command to validate URDF files
- Test in simulation before deploying to physical robots
- Verify kinematic solutions match expectations

## Summary

- **URDF** defines robot structure, kinematics, and dynamics in XML format
- **Links** represent rigid components of the robot body
- **Joints** connect links and define their relative motion
- **Transmissions** map actuators to joints for control
- **Materials and properties** define visual and collision characteristics
- **Humanoid robots** have specific kinematic chains for arms, legs, and torso
- **URDF bridges** the gap between software models and physical robots
- **Best practices** ensure accurate and efficient robot modeling

Understanding URDF is crucial for creating AI agents that can effectively control humanoid robots and understand their physical constraints and capabilities.

## See Also

- Review the fundamental [ROS 2 Core Concepts](./core-concepts.md) for communication patterns
- Learn how to implement these concepts in Python with [rclpy](./python-agents-rclpy.md)