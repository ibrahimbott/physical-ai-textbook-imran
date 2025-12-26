---
title: "Module 7: Humanoid Design"
description: "Kinematics, Dynamics, and the Physics of Walking"
module: 7
duration: "8-12 hours"
prerequisites: "Classical Mechanics, Linear Algebra"
objectives:
  - Solve Forward and Inverse Kinematics for a limb chain.
  - Understand the Zero Moment Point (ZMP) stability criterion.
  - Analyze the phases of a bipedal gait cycle.
  - Explore actuator placement and mass distribution trade-offs.
---

# Module 7: Humanoid Design

Building a humanoid is fighting gravity. Every gram of weight in the hand requires torque in the shoulder, which requires battery in the torso.

<img src="/img/module7_zmp.png" class="module-header-image" />

## 1. Kinematics

How do we know where the hand is?

### Forward Kinematics (FK)
Given joint angles $\theta_1, \theta_2 \dots$, calculate the hand's $(x, y, z)$.
This is simple chain multiplication of transformation matrices (DH Parameters).

### Inverse Kinematics (IK)
Given the desire "Put hand at $(x, y, z)$", calculate the required angles.
* This is hard. Solutions are often non-linear and not unique (elbow up vs. elbow down).
* We use numerical solvers (Jacobian pseudoinverse) or geometric analysis.

## 2. Stability: The ZMP Criterion

For a static object, stability means the Center of Mass (CoM) projection is inside the feet.
For a *dynamic* walker, we use the **Zero Moment Point (ZMP)**.

> **ZMP Rule**: The robot will not tip over if the ZMP lies strictly within the support polygon (the shape formed by the feet on the ground).

If the robot accelerates forward effectively, it "pushes" the ZMP backward. To stay stable, it must lean forward (like a sprinter).

## 3. The Bipedal Gait Cycle

Walking is "controlled falling".

1.  **Double Support (20%)**: Both feet on ground. Transferring weight.
2.  **Single Support (80%)**: One foot planted (Stance), one foot moving (Swing).
    - **Stance Leg**: Bears the load. Knee must not buckle.
    - **Swing Leg**: Must actuate quickly to catch the fall. Also must lift specifically to avoid reliable toe-stubbing.

## 4. Mechanical Design Tips

- **Distal Mass**: Keep heavy motors near the body (proximal). A heavy hand requires huge shoulder motors (high Moment of Inertia). Use cable drives or linkages to move mass inward.
- **Series Elastic Actuators (SEA)**: Adding a spring between motor and load improves shock resistance (walking impact) and allows force sensing.
