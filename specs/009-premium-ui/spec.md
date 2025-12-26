# Spec 009: Premium Chatbot UI Overhaul

## 1. Background
The user requested: "Same to same better UI, premium designs, glassmorphism, logic same but look improved".
Goal: Replace the missing/broken frontend with a high-fidelity, premium web interface served directly by the backend.

## 2. Requirements

### Functional
*   **FR-001**: System MUST serve `index.html` at root `/`.
*   **FR-002**: System MUST preserve chat history during the session.
*   **FR-003**: System MUST display "Thinking..." animations while waiting for API.
*   **FR-004**: System MUST render Markdown (bold, lists, code) client-side.

### Visual
*   **VR-001**: Color Palette: Deep Slate (`#0f172a`), Indigo Primary (`#6366f1`), Pink Secondary (`#ec4899`).
*   **VR-002**: Typography: 'Outfit' (Google Fonts).
*   **VR-003**: Effects: `backdrop-filter: blur(10px)`.

## 3. User Stories
*   **US-1**: As a user, I want a modern, "wow" interface so that studying feels engaging.
*   **US-2**: As a student, I want to see code blocks and source citations clearly.
*   **US-3**: As a mobile user, I want a responsive sidebar.
