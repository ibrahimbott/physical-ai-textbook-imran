// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1 - The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2-nervous-system/index',
        'module-1-ros2-nervous-system/core-concepts',
        'module-1-ros2-nervous-system/python-agents-rclpy',
        'module-1-ros2-nervous-system/humanoid-urdf',
      ],
    },
    {
      type: 'category',
      label: 'Module 2 - The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin-simulation/index',
        'module-2-digital-twin-simulation/physics-simulation-gazebo',
        'module-2-digital-twin-simulation/high-fidelity-unity',
        'module-2-digital-twin-simulation/simulated-sensors',
      ],
    },
    {
      type: 'category',
      label: 'Module 3 - Simulation & Digital Twins',
      items: [
        'simulation/module-2-simulation',
        'simulation/digital-twins',
        'simulation/gazebo-unity',
      ],
    },
    {
      type: 'category',
      label: 'Module 4 - Hardware Basics',
      items: [
        'hardware-basics/module-3-hardware',
      ],
    },
    {
      type: 'category',
      label: 'Module 5 - VLA Systems',
      items: [
        'vla-systems/module-4-vla-foundations',
        'vla-systems/vla-vision',
        'vla-systems/vla-language',
        'vla-systems/vla-action',
        'vla-systems/vla-hands-on-basic',
      ],
    },
    {
      type: 'category',
      label: 'Module 6 - Advanced AI Control',
      items: [
        'advanced-ai-control/module-5-advanced-ai',
      ],
    },
    {
      type: 'category',
      label: 'Module 7 - Humanoid Design',
      items: [
        'humanoid-design/module-6-humanoid-design',
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      items: [
        'appendix/glossary',
        'appendix/references',
        'appendix/resources',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
