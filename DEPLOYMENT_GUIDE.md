# 🚀 Deployment Guide: Physical AI Textbook

You asked: *"How do I deploy the whole website including both? How will backend start with frontend together?"*

## The Solution: **Vercel**

Vercel is the best platform for this because it natively understands how to run both your **Docusaurus Frontend** and your **Python Backend** from a single repository.

### How it works "Together"
On Vercel, they don't run as two separate "servers" like they do on your laptop. Instead:
1.  **The Frontend (Book):** Vercel builds your Docusaurus site into static HTML/CSS/JS files and puts them on a worldwide CDN. They are "always on."
2.  **The Backend (Chatbot):** Vercel turns your Python code into **Serverless Functions**. These sleep until someone asks the Chatbot a question. When the request comes to `/api/chat`, Vercel instantly wakes up the backend, answers, and goes back to sleep.
3.  **The Glue (`vercel.json`):** This file tells Vercel: "Send anything starting with `/api/` to Python, and everything else to the Book."

---

## ✅ Step 1: Verify Your Code (Already Done by Agent)

I have already configured everything for you:
*   [x] **`vercel.json`**: Configured to route traffic correctly.
*   [x] **`api/index.py`**: The entry point for Vercel's Python runtime.
*   [x] **`requirements.txt`**: Placed in the root so Vercel can install dependencies.
*   [x] **`.gitignore`**: Updated to keep your repo clean.

## 🚀 Step 2: Push to GitHub

1.  Open your terminal.
2.  Run these commands to save everything and push to GitHub:
    ```bash
    git add .
    git commit -m "Ready for Vercel Deployment"
    git push origin main
    ```

## 🌍 Step 3: Connect to Vercel

1.  Go to [**vercel.com**](https://vercel.com) and Log In.
2.  Click **"Add New..."** -> **"Project"**.
3.  Select your GitHub Repository (`Physical-AI-TextBook`).
4.  **Configure Project:**
    *   **Framework Preset:** Vercel should auto-detect "Docusaurus" or "Other". If not, verify strict settings don't block it. Default is usually fine.
    *   **Root Directory:** Leave as `./` (Root).

## 🔑 Step 4: Add Environment Variables (CRITICAL)

**Before clicking Deploy**, you MUST add these variables in the "Environment Variables" section of the Vercel setup screen. If you miss these, the Chatbot will say "Connection Error".

| Variable Name | Value |
| :--- | :--- |
| `GOOGLE_API_KEY` | *(Your Gemini API Key)* |
| `QDRANT_URL` | *(Your Qdrant Cluster URL)* |
| `QDRANT_API_KEY` | *(Your Qdrant API Key)* |
| `COHERE_API_KEY` | *(Your Cohere API Key)* |
| `COLLECTION_NAME` | `physical_ai_textbook` |
| `MODEL_NAME` | `gemini-flash-latest` |
| `NODE_OPTIONS` | `--max-old-space-size=6144` |

> **Note:** The `NODE_OPTIONS` variable is already in your `vercel.json`, but adding it to the dashboard is a safe backup to prevent memory errors during build.

## 🎉 Step 5: Deploy

1.  Click **"Deploy"**.
2.  Wait for the build to finish (usually 2-3 minutes).
3.  **Success!** You will get a live URL (e.g., `https://physical-ai-textbook.vercel.app`).

### Testing the Live Site
1.  Go to your new URL.
2.  Open the Chatbot.
3.  Say "Hello".
    *   It should reply instantly.
    *   If it errors, go to **Vercel Dashboard > Deployments > [Click Verification] > Functions** to see the logs.
