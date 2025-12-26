---
title: "Module 5: VLA Systems"
description: "Vision-Language-Action Models: The Brain of the Humanoid"
module: 5
duration: "6-8 hours"
prerequisites: "Python, PyTorch, Transformers basics"
objectives:
  - Explain the architecture of VLA models (Vision Encoder + LLM + Action Head).
  - Understand how images and text are tokenized into a shared embedding space.
  - Review state-of-the-art models: RT-2, Octo, OpenVLA.
  - Design a prompt strategy for controlling a robot via natural language.
---

# Module 5: Vision-Language-Action (VLA) Systems

The era of "hard-coded" robotics is ending. **Vision-Language-Action (VLA)** models allow us to speak to robots and have them understand the visual world to generate low-level actions.

<img src="/img/module5_vla.png" class="module-header-image" />

## 1. The Architecture of Agency

A VLA model is essentially a Multimodal LLM trained to output "action tokens" alongside text.

```mermaid
graph LR
    A[Camera RGB] --> B(Vision Encoder<br/>SigLIP/ViT);
    C[User Command<br/>'Pick up the apple'] --> D(Tokenizer);
    B --> E[Transformer Backbone<br/>Llama / Vicuna];
    D --> E;
    E --> F{Output};
    F -->|Text| G[Reasoning:<br/>'I see an apple...'];
    F -->|Actions| H[Joint Velocities<br/>[0.1, -0.5, 0.2...]];
```

### Components
1.  **Vision Encoder**: Compresses the high-dimensional image into compact embeddings (tokens). Popular choices: *SigLIP*, *CLIP*, *DINOv2*.
2.  **LLM Backbone**: The "brain" that reasons. It takes mixed sequences of text and image tokens.
3.  **Action Head / Tokenizer**: The magic glue. We discretize continuous actions (move arm 5cm) into discrete tokens (vocabulary IDs 32000-32256).

## 2. Case Studies

### RT-2 (Robotics Transformer 2)
Google DeepMind's RT-2 demonstrated that a model trained on internet-scale web data can perform robotic tasks it never practiced, due to semantic understanding.
- *Key Insight*: Co-training on web data (VQA) and robot data (trajectories).

### Octo & OpenVLA
Open-source alternatives that allow us to run VLAs on our own hardware (like the Jetson Orin!).
- **OpenVLA**: Built on Llama 3 / Vicuna. Fine-tuned on the *Open X-Embodiment* dataset (millions of robot trajectories).

## 3. Hands-on: Prompting a Robot

Unlike ChatGPT, a VLA needs a structured environment context.

**System Prompt Example:**
```text
You are a robotic assistant controlling a Franka Emika arm. 
The input image shows the current workspace.
Your output must be a sequence of 7 joint velocities in the format: <ACT> v1 v2 v3 v4 v5 v6 v7 </ACT>.
Goal: Pick up the red soda can.
```

## 4. Challenges
- **Latency**: LLMs are slow (tokens per second). Robots need 50Hz control loops.
- **Inference**: Running a 7B parameter model on edge requires quantization (4-bit/8-bit).
