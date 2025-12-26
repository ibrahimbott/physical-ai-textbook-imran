# Quickstart Guide: Module 2 — The Digital Twin (Gazebo & Unity)

**Feature**: Module 2 — The Digital Twin (Gazebo & Unity)
**Date**: 2025-12-20
**Status**: Implementation Ready

## Prerequisites

- Node.js 18+ installed (for Docusaurus)
- npm or yarn package manager
- Git for version control
- Basic understanding of simulation concepts
- Familiarity with documentation tools

## Setup Instructions

### 1. Navigate to Frontend Book Directory

```bash
# Navigate to the existing frontend-book directory
cd D:\Python\GIAI\Physical-AI\Physical-AI-TextBook\frontend-book
```

### 2. Create Module Directory Structure

```bash
# Create docs directory structure for the new module
mkdir -p docs/module-2-digital-twin-simulation
```

### 3. Create Module Content Files

Create the three chapter files as specified:
- `docs/module-2-digital-twin-simulation/index.md` - Main module overview
- `docs/module-2-digital-twin-simulation/physics-simulation-gazebo.md` - Gazebo physics simulation
- `docs/module-2-digital-twin-simulation/high-fidelity-unity.md` - Unity environments
- `docs/module-2-digital-twin-simulation/simulated-sensors.md` - Sensor simulation

### 4. Update Sidebar Navigation

Update `sidebars.js` to include the new module by adding it to the existing sidebar structure:

```javascript
// sidebars.js - Add to existing sidebar configuration
{
  type: 'category',
  label: 'Module 2 - The Digital Twin (Gazebo & Unity)',
  items: [
    'module-2-digital-twin-simulation/index',
    'module-2-digital-twin-simulation/physics-simulation-gazebo',
    'module-2-digital-twin-simulation/high-fidelity-unity',
    'module-2-digital-twin-simulation/simulated-sensors',
  ],
}
```

### 5. Run Development Server

```bash
# Start the development server to verify changes
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
5. **Navigation Test**: Verify sidebar navigation works correctly for new module

## Deployment

```bash
# Build the site for production
npm run build

# The build will be available in the build/ directory
# Ready for deployment to GitHub Pages
```

The site will be automatically deployed to GitHub Pages if properly configured in your repository settings.