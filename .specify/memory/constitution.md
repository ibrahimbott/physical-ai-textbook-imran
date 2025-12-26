<!-- SYNC IMPACT REPORT
Version change: N/A → 1.0.0
Modified principles: N/A (initial creation)
Added sections: All sections (initial creation)
Removed sections: N/A
Templates requiring updates: N/A (initial file)
Follow-up TODOs: None
-->

# Project Constitution

**Project:** AI-Spec-Driven Book with Embedded RAG Chatbot

**Version:** 1.0.0

**Ratification Date:** 2025-12-20

**Last Amended Date:** 2025-12-20

## Core Principles

### 1. Spec-first Development
All implementation must be preceded by a formal specification. No code should be written without an approved spec that defines the feature's requirements, acceptance criteria, and validation approach.

_Rationale: Ensures clarity of purpose, reduces rework, and enables distributed development with predictable outcomes._

### 2. Accuracy Aligned with Official Documentation
All content and code must be verified against official documentation and authoritative sources. No implementation should rely on assumptions or undocumented behavior.

_Rationale: Maintains high quality standards and ensures reproducibility of results across different environments._

### 3. Clarity for Software Engineers and AI Practitioners
Content must be accessible to both traditional software engineers and AI practitioners. Documentation should be clear, with examples that demonstrate practical application.

_Rationale: Enables broader adoption and reduces barrier to entry for diverse audiences._

### 4. Full Reproducibility of Content, Code, and Deployment
Every aspect of the project must be fully reproducible. Build processes, deployments, and examples must work consistently across different environments.

_Rationale: Ensures trust in the material and enables others to replicate results reliably._

### 5. Reusable Intelligence via Agents and Subagents
Intelligence should be encapsulated in reusable components that can be leveraged across different contexts. Agents and subagents must have clearly defined scopes and interfaces.

_Rationale: Reduces future development effort and creates compounding value from initial investments._

## Book Standards

### 1. Docusaurus-Based Documentation
All book content must be written using Docusaurus with Markdown formatting. This ensures consistent presentation and deployment capabilities.

_Rationale: Provides professional-grade documentation tools with modern features and easy maintenance._

### 2. GitHub Pages Deployment
The book must be deployable to and hosted on GitHub Pages, ensuring public accessibility and version control integration.

_Rationale: Enables free hosting with integrated version control and broad accessibility._

### 3. Structured Chapter Format
Each chapter must include clear objectives, practical examples, and comprehensive summaries to facilitate learning.

_Rationale: Creates consistent learning experiences and enables self-paced study._

### 4. Appropriate Reading Level
Content must maintain a Flesch-Kincaid grade level between 10-12 to ensure accessibility without sacrificing technical depth.

_Rationale: Balances technical accuracy with comprehension for the target audience._

### 5. Original Content Requirement
All content must be original with zero tolerance for plagiarism. Proper attribution must be given where external concepts are referenced.

_Rationale: Maintains academic integrity and avoids copyright issues._

## RAG Chatbot Standards

### 1. Context-Restricted Operation
The RAG chatbot must only answer questions based on: (1) full book content, or (2) user-selected text. It must refuse out-of-context queries.

_Rationale: Maintains accuracy and prevents hallucination of information outside the established knowledge base._

### 2. Defined Technology Stack
The chatbot must use the specified stack: FastAPI, OpenAI Agents/ChatKit SDKs, Neon Serverless Postgres, and Qdrant Cloud (Free Tier).

_Rationale: Ensures compatibility, cost-effectiveness, and maintainability within defined constraints._

### 3. Explicit Source Citation
All responses must include explicit chunking, retrieval, and citation of sources to maintain transparency.

_Rationale: Enables verification of responses and maintains trust in the system's accuracy._

## Agents & Subagents Standards

### 1. Clear Role Scoping
Each agent and subagent must have clearly defined roles with no overlap or redundancy in functionality.

_Rationale: Prevents confusion and ensures efficient resource utilization._

### 2. Reusable Agent Skills
Agent skills must be designed for reuse across different contexts with well-defined inputs, outputs, and failure modes.

_Rationale: Maximizes return on development investment and promotes consistency._

## Validation & Testing Requirements

### 1. Build Verification
Docusaurus build must pass without errors, and GitHub Pages deployment must succeed consistently.

_Rationale: Ensures deliverable quality and deployment reliability._

### 2. Deterministic Retrieval
RAG retrieval must be deterministic and traceable, with consistent results for identical queries.

_Rationale: Enables testing and verification of system behavior._

### 3. Context Enforcement
Context-restricted answering must be enforced programmatically to prevent out-of-scope responses.

_Rationale: Maintains system integrity and accuracy of responses._

## Constraints

### 1. No Undocumented Assumptions
Implementation must not rely on undocumented assumptions or implicit dependencies.

_Rationale: Ensures maintainability and reduces technical debt._

### 2. No Hallucinated Content
Systems must not generate content that is not based on documented facts or established knowledge.

_Rationale: Maintains accuracy and prevents propagation of incorrect information._

### 3. Free-Tier Dependency Limitation
Core functionality must not depend on paid-only services, though optional upgrades are permitted.

_Rationale: Ensures accessibility and avoids vendor lock-in for core functionality._

## Governance

### Amendment Procedure
Changes to this constitution require explicit approval through the project's established review process. Minor clarifications may be made with team consensus, while major changes require formal proposal and review.

### Versioning Policy
- MAJOR: Backward incompatible governance/principle removals or redefinitions
- MINOR: New principle/section added or materially expanded guidance
- PATCH: Clarifications, wording, typo fixes, non-semantic refinements

### Compliance Review
Regular reviews must be conducted to ensure project activities align with constitutional principles.

## Success Criteria

### 1. Public Availability
A public book site must be available and accessible to users worldwide.

### 2. Functional Chatbot
The embedded chatbot must function according to the specified requirements and standards.

### 3. Reproducible Systems
All specifications and systems must be reproducible by third parties following the provided documentation.

### 4. Demonstrable Intelligence Reuse
Reusable intelligence must demonstrably reduce future work effort compared to de novo development approaches.