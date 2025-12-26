# Spec 012: Content Modules Integration

## 1. Background
The textbook content was fragmented or missing. We needed to integrate the core educational material into the Docusaurus structure.

## 2. Requirements
*   **Structure**: 7 distinct modules ranging from ROS 2 (Intro) to Humanoid Design (Advanced).
*   **Navigation**: Sidebar must allow linear progression.
*   **Content**: Each module needs a meaningful introduction, learning objectives, and placeholders for deeper technical content.
*   **Appendix**: Glossary, References, and Resources sections.

## 3. Strategy
*   **Sidebar**: Flatten the directory structure where possible for cleaner URLs (removed numerical prefixes from slugs).
*   **Intro**: Convert `intro.md` to a "Welcome" page.
