---
sidebar_label: 'ROS 2 Core Concepts'
sidebar_position: 2
---

# ROS 2 Core Concepts

## Learning Objectives

- Understand the fundamental components of ROS 2: nodes, topics, and services
- Explain how ROS 2 functions as a distributed nervous system for robots
- Describe data flow and message passing mechanisms in ROS 2 architecture
- Identify the role of parameters and actions in ROS 2 systems

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is not an operating system but rather a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

ROS 2 serves as the distributed nervous system for robots, enabling different software components to communicate with each other regardless of the programming language they are implemented in or the computer system they run on.

## Nodes

A **node** is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 system. Each node represents a specific function or capability within the robot system.

### Node Characteristics

- **Independent processes**: Nodes run as separate processes that communicate with other nodes through messages
- **Single responsibility**: Each node typically handles one specific task or function
- **Language agnostic**: Nodes can be written in different programming languages (C++, Python, etc.) and still communicate
- **Process isolation**: Failure of one node doesn't necessarily affect others

### Node Lifecycle

Nodes in ROS 2 have a well-defined lifecycle that allows for more sophisticated management:

- **Unconfigured**: Initial state after creation
- **Inactive**: Configured but not active
- **Active**: Fully operational and running
- **Finalized**: Clean shutdown state

## Topics and Message Passing

**Topics** are named buses over which nodes exchange messages in a publish/subscribe pattern. This is the primary method of asynchronous communication in ROS 2.

### Publish/Subscribe Pattern

- **Publisher**: A node that sends messages to a topic
- **Subscriber**: A node that receives messages from a topic
- **Anonymous**: Publishers and subscribers are unaware of each other's existence
- **Decoupled**: Timing between publishers and subscribers is independent

### Quality of Service (QoS)

ROS 2 provides Quality of Service settings that allow you to fine-tune communication characteristics:

- **Reliability**: Reliable vs. best-effort delivery
- **Durability**: Keep last message vs. transient local
- **History**: Number of messages to keep in the queue
- **Deadline**: Maximum time between messages
- **Liveliness**: How to determine if a participant is alive

## Services

**Services** provide a request/response communication pattern between nodes for synchronous operations. This is useful when you need to wait for a response before continuing.

### Service Characteristics

- **Synchronous**: The client waits for a response from the server
- **Request/Response**: Defined message types for both request and response
- **Bidirectional**: Both request and response message types are specified
- **Blocking**: The client blocks until the service call completes

### When to Use Services vs Topics

Use services when:
- You need a response to a specific request
- The operation has a clear start and end
- You need to wait for the result before proceeding

Use topics when:
- You want to broadcast information continuously
- You don't need a response
- You want to decouple timing between nodes

## Parameters

**Parameters** are configuration values that nodes can set and get. They provide a way to configure nodes without recompiling the code.

### Parameter Features

- **Dynamic**: Parameters can be changed at runtime
- **Typed**: Support for various data types (int, float, string, bool, lists)
- **Hierarchical**: Parameters can be organized in a namespace hierarchy
- **Persistent**: Can be saved and loaded from configuration files

## Actions

**Actions** are used for long-running tasks that may provide feedback and can be canceled. They combine the benefits of services and topics.

### Action Components

- **Goal**: The request sent to start an action
- **Feedback**: Continuous updates on the progress of the action
- **Result**: The final outcome of the action when completed

### Action Characteristics

- **Long-running**: Designed for tasks that take a significant amount of time
- **Cancellable**: Actions can be canceled before completion
- **Feedback-enabled**: Provides continuous feedback during execution
- **Stateful**: Maintains state throughout the action lifecycle

## ROS 2 as a Distributed Nervous System

ROS 2 architecture mirrors biological nervous systems in several ways:

### Distributed Processing

Like the nervous system, ROS 2 distributes processing across multiple nodes rather than having a single central processor. Different nodes handle different functions (sensory input, motor control, decision making).

### Communication Patterns

- **Sensory pathways**: Analogous to topics, where sensory information flows from sensors to processing nodes
- **Motor pathways**: Similar to service calls or action commands that control actuators
- **Feedback loops**: Continuous monitoring and adjustment like biological feedback systems

### Real-time vs Non-real-time Communication

ROS 2 supports different communication requirements:
- **Real-time critical**: Time-sensitive communication for safety or performance
- **Non-real-time**: Less time-critical communication for logging, visualization, etc.

## Practical Example: Simple Robot System

Consider a simple robot with the following nodes:
- **Sensor node**: Publishes laser scan data to `/scan` topic
- **Navigation node**: Subscribes to sensor data and publishes velocity commands to `/cmd_vel` topic
- **Motor controller node**: Subscribes to velocity commands and controls the physical motors

This distributed architecture allows each component to be developed, tested, and maintained independently while maintaining clear communication interfaces.

## Summary

- **Nodes** are the fundamental building blocks that perform specific functions
- **Topics** enable asynchronous communication through publish/subscribe patterns
- **Services** provide synchronous request/response communication
- **Parameters** allow for dynamic configuration of nodes
- **Actions** handle long-running tasks with feedback and cancellation
- ROS 2 functions as a distributed nervous system, enabling modular and scalable robot architectures

Understanding these core concepts is essential for designing effective communication patterns between AI agents and robot systems.

## See Also

- Learn how to implement these concepts in Python with [rclpy](./python-agents-rclpy.md)
- Understand how to represent robot structure with [URDF](./humanoid-urdf.md)