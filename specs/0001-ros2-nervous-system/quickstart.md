# Quickstart Guide: Module 1 — The Robotic Nervous System (ROS 2)

**Feature**: Module 1 — The Robotic Nervous System (ROS 2)
**Date**: 2025-12-20
**Status**: Implementation Ready

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Git for version control
- Basic understanding of Python (for rclpy content)
- Familiarity with documentation tools

## Setup Instructions

### 1. Initialize Docusaurus Project

```bash
# Navigate to project root
cd D:\Python\GIAI\Physical-AI\Physical-AI-TextBook

# Create frontend-book directory if it doesn't exist
mkdir frontend-book
cd frontend-book

# Initialize Docusaurus
npm init docusaurus@latest .

# Select "classic" template when prompted
```

### 2. Configure Docusaurus Site

Edit `docusaurus.config.js` to set up the site configuration:

```javascript
// docusaurus.config.js
module.exports = {
  title: 'Physical AI Textbook',
  tagline: 'Connecting AI Agents to Physical Robot Systems',
  favicon: 'img/favicon.ico',

  url: 'https://your-github-username.github.io',
  baseUrl: '/Physical-AI-TextBook/',

  organizationName: 'your-org', // Usually your GitHub org/user name
  projectName: 'Physical-AI-TextBook', // Usually your repo name

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-org/Physical-AI-TextBook/tree/main/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
};
```

### 3. Create Module Directory Structure

```bash
# Create docs directory structure for the module
mkdir -p docs/module-1-ros2-nervous-system

# Create the main module index file
touch docs/module-1-ros2-nervous-system/index.md
```

### 4. Create Module Content Files

Create the three chapter files as specified:
- `docs/module-1-ros2-nervous-system/core-concepts.md`
- `docs/module-1-ros2-nervous-system/python-agents-rclpy.md`
- `docs/module-1-ros2-nervous-system/humanoid-urdf.md`

### 5. Configure Sidebar Navigation

Update `sidebars.js` to include the new module:

```javascript
// sidebars.js
module.exports = {
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
  ],
};
```

### 6. Run Development Server

```bash
# Start the development server
npm start

# The site will be available at http://localhost:3000
```

## Content Creation Guidelines

### For Each Chapter:

1. **Include Learning Objectives** at the beginning
2. **Provide Conceptual Examples** without full implementations
3. **Add Summaries** at the end of each chapter
4. **Maintain Flesch-Kincaid Grade Level 10-12**
5. **Use Original Content** (0% plagiarism)

### Chapter Structure Template:

```markdown
---
id: page-id
title: Chapter Title
sidebar_label: Chapter Title
---

## Learning Objectives

- Objective 1
- Objective 2
- Objective 3

## Content

[Main content with concepts, explanations, and examples]

## Summary

- Key takeaway 1
- Key takeaway 2
- Key takeaway 3
```

## Validation Steps

1. **Build Verification**: Run `npm run build` to ensure site builds without errors
2. **Link Checking**: Verify all internal links work correctly
3. **Content Review**: Ensure all content meets educational objectives
4. **Style Check**: Confirm content maintains grade level 10-12 readability

## Deployment

```bash
# Build the site for production
npm run build

# Deploy to GitHub Pages (if configured)
git add .
git commit -m "Add Module 1 - The Robotic Nervous System (ROS 2)"
git push origin 001-ros2-nervous-system
```

The site will be automatically deployed to GitHub Pages if properly configured in your repository settings.