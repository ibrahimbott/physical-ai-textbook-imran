# Physical AI Textbook

Welcome to the Physical AI Textbook, a comprehensive resource for connecting AI agents to physical robot systems. This textbook focuses on ROS 2 as middleware for humanoid robot control and communication between AI agents and physical robot systems.

## Current Modules

- **Module 1: The Robotic Nervous System (ROS 2)** - Core concepts, Python agents with rclpy, and humanoid structure with URDF
- **Module 2: The Digital Twin (Gazebo & Unity)** - Physics-based simulation, virtual environments, and sensor simulation for digital twins of humanoid robots

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
