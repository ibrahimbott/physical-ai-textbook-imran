---
sidebar_label: 'Python Agents with rclpy'
sidebar_position: 3
---

# Python Agents with rclpy

## Learning Objectives

- Understand the rclpy library and its role in ROS 2 Python development
- Create ROS 2 nodes in Python using rclpy
- Implement publisher and subscriber patterns in Python
- Build service clients and servers using rclpy
- Apply best practices for connecting AI logic to robot controllers

## Introduction to rclpy

**rclpy** is the Python client library for ROS 2. It provides a Python API that allows developers to create ROS 2 nodes, publish and subscribe to topics, provide and use services, and manage parameters. This library enables Python-based AI agents to communicate effectively with robot systems.

### Key Features of rclpy

- **Node management**: Create and manage ROS 2 nodes in Python
- **Communication patterns**: Support for topics, services, and actions
- **Parameter handling**: Dynamic configuration of nodes
- **Lifecycle management**: Proper initialization and cleanup
- **Threading support**: Asynchronous and multi-threaded operations

## Creating ROS 2 Nodes with rclpy

A ROS 2 node in Python is created by subclassing `rclpy.node.Node` and implementing the required functionality.

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        # Initialize node components here
        self.get_logger().info('MyRobotNode initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle Management

Proper lifecycle management ensures resources are properly allocated and released:

- **Initialization**: Set up publishers, subscribers, services, and parameters
- **Execution**: Process callbacks and perform node functions
- **Shutdown**: Clean up resources and gracefully terminate

## Publisher and Subscriber Patterns

### Publishers

Publishers send messages to topics, enabling asynchronous communication:

```python
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Subscribers

Subscribers receive messages from topics and process them in callbacks:

```python
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

### Quality of Service (QoS) Configuration

QoS settings can be configured for publishers and subscribers:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Apply to publisher
publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

## Service Clients and Servers

### Service Servers

Service servers provide synchronous request/response communication:

```python
from example_interfaces.srv import AddTwoInts

class ServiceServerNode(Node):
    def __init__(self):
        super().__init__('service_server_node')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response
```

### Service Clients

Service clients call services provided by other nodes:

```python
from example_interfaces.srv import AddTwoInts

class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client_node')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Parameter Handling in rclpy

Parameters allow dynamic configuration of nodes:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('threshold', 0.5)

        # Get parameter values
        my_param = self.get_parameter('my_parameter').value
        threshold = self.get_parameter('threshold').value

        # Parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f'Threshold updated to: {param.value}')
        return SetParametersResult(successful=True)
```

## Best Practices for Asynchronous Processing

### Threading Models

rclpy supports both single-threaded and multi-threaded execution:

```python
# Single-threaded executor (default)
rclpy.spin(node)

# Multi-threaded executor
from rclpy.executors import MultiThreadedExecutor

executor = MultiThreadedExecutor()
executor.add_node(node)
executor.spin()
```

### Callback Handling

Proper callback design ensures responsive nodes:

```python
import threading

class ThreadedNode(Node):
    def __init__(self):
        super().__init__('threaded_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.threaded_callback,
            10)

    def threaded_callback(self, msg):
        # Run intensive processing in a separate thread
        thread = threading.Thread(target=self.process_data, args=(msg,))
        thread.daemon = True
        thread.start()

    def process_data(self, msg):
        # Time-intensive processing here
        self.get_logger().info(f'Processed: {msg.data}')
```

## Error Handling and Recovery Patterns

Robust error handling is crucial for reliable robot systems:

```python
class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        try:
            self.setup_components()
        except Exception as e:
            self.get_logger().error(f'Failed to setup components: {e}')
            raise

    def setup_components(self):
        # Setup publishers, subscribers, etc.
        pass

    def safe_publish(self, publisher, msg):
        try:
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish message: {e}')
```

## Connecting AI Logic to Robot Controllers

The primary goal of using rclpy in AI applications is to bridge AI decision-making with robot control systems.

### AI-Agent Integration Pattern

```python
import numpy as np
from geometry_msgs.msg import Twist

class AIAgentNode(Node):
    def __init__(self):
        super().__init__('ai_agent_node')

        # Publisher for sending commands to robot
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for sensor data
        self.sensor_subscription = self.create_subscription(
            LaserScan, '/scan', self.sensor_callback, 10)

        # Store sensor data for AI processing
        self.sensor_data = None

        # Timer for AI decision making
        self.ai_timer = self.create_timer(0.1, self.ai_decision_callback)

    def sensor_callback(self, msg):
        # Process incoming sensor data
        self.sensor_data = msg.ranges  # Example: laser scan ranges

    def ai_decision_callback(self):
        if self.sensor_data is not None:
            # Apply AI logic to make decisions
            velocity_command = self.make_ai_decision(self.sensor_data)

            # Publish command to robot
            self.cmd_vel_publisher.publish(velocity_command)

    def make_ai_decision(self, sensor_data):
        # AI logic implementation
        cmd = Twist()
        # Example: simple obstacle avoidance
        min_distance = min(sensor_data) if sensor_data else float('inf')

        if min_distance > 1.0:  # No obstacles nearby
            cmd.linear.x = 0.5  # Move forward
            cmd.angular.z = 0.0
        else:  # Obstacle detected
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn to avoid

        return cmd
```

## Performance Considerations

### Message Rate Optimization

Consider the appropriate message rate for your application:

- **High-frequency**: Control loops, typically 50-200 Hz
- **Medium-frequency**: Sensor processing, typically 10-50 Hz
- **Low-frequency**: Planning, monitoring, typically 1-10 Hz

### Memory Management

Be mindful of memory usage in long-running nodes:

```python
# Limit history of processed data
class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')
        self.max_history = 100
        self.history = []

    def process_callback(self, msg):
        self.history.append(msg)
        if len(self.history) > self.max_history:
            self.history.pop(0)  # Remove oldest entry
```

## Summary

- **rclpy** provides the Python API for ROS 2 development
- **Nodes** are created by subclassing `rclpy.node.Node`
- **Publishers and subscribers** enable asynchronous communication
- **Services** provide synchronous request/response patterns
- **Parameters** allow dynamic configuration of nodes
- **Best practices** include proper error handling, threading, and performance optimization
- **AI integration** involves bridging decision-making logic with robot control systems

Using rclpy effectively allows Python-based AI agents to communicate seamlessly with robot systems, enabling sophisticated robot behaviors driven by artificial intelligence.

## See Also

- Review the fundamental [ROS 2 Core Concepts](./core-concepts.md) these implementations are based on
- Learn how to represent robot structure with [URDF](./humanoid-urdf.md) for complete robot modeling