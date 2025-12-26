# Plan: Premium Chatbot UI Overhaul

## 1. Frontend Implementation
-   [x] Create glassmorphism CSS variables in `index.html` (embedded styles or separate).
-   [x] Implement responsive sidebar logic (hamburger menu).
-   [x] Add "typing..." animation styles.

## 2. Backend Integration
-   [x] Ensure FastAPI serves `static/index.html` at `/`.
-   [x] Verify `/chat` endpoint accepts JSON and returns Markdown.

## 3. Deployment
-   [x] Bundle vanilla js/css into `backend/static`.
