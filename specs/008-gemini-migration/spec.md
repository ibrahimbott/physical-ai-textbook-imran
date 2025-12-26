# Feature Specification: Migration to Google Gemini (Via OpenAI SDK)

**Feature Branch**: `008-gemini-migration`
**Created**: 2025-12-26
**Status**: Implemented
**Input**: User request: "Fix logic, OpenAI Agent SDK not functioning, switch to Gemini free tier"

Goal: Migrate the existing RAG Agent backend to use Google Gemini models (via OpenAI SDK compatibility) to fix availability issues and reduce costs.

Success criteria:
- Replace OpenAI GPT-4 configurations with Google Gemini models.
- Implement OpenAI SDK `base_url` override for Gemini.
- Backend successfully connects to Gemini 2.0 Flash (or available variant).
- "Agent" logic (function calling) is preserved and works with Gemini.

Constraints:
- Must use existing OpenAI SDK code structure where possible.
- Must handle "Model Not Found" errors gracefully by fallback.
- Must not break existing Qdrant retrieval.

Not building:
- New retrieval logic (reusing existing).
- New embedding provider (keeping Cohere).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Backend Connectivity (Priority: P0)

As a developer, I want the backend to connect to Gemini so that I can use a free/available LLM without OpenAI credit issues.

**Why this priority**: The previous system was broken due to unavailable OpenAI models.

**Independent Test**:
- Set `GOOGLE_API_KEY`.
- Run `test_setup.py` (updated).
- Verify successful connection to `gemini-2.0-flash`.

**Acceptance Scenarios**:
1. **Given** a valid Google API Key, **When** the system starts, **Then** it authenticates with `generativelanguage.googleapis.com`.
2. **Given** a request for `gemini-1.5-flash` that fails (404), **When** the system retries, **Then** it falls back to `gemini-2.0-flash` or `gemini-flash-latest`.

---

### User Story 2 - Function Calling with Gemini (Priority: P1)

As a user, I want the AI to still be able to "search the textbook" so that I get grounded answers.

**Why this priority**: RAG depends on the LLM's ability to call the retrieval tool.

**Acceptance Scenarios**:
1. **Given** a question about the textbook, **When** the agent processes it, **Then** it generates a `retrieve_knowledge` tool call.
2. **Given** tool outputs, **When** the agent generates the final answer, **Then** it uses the provided context.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST use `GOOGLE_API_KEY` for LLM authentication.
- **FR-002**: System MUST override OpenAI client `base_url` to Google's endpoint.
- **FR-003**: System MUST support `gemini-2.0-flash` as the primary model.
- **FR-004**: System MUST implement model availability checks/fallback.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `test_setup.py` passes with Gemini configuration.
- **SC-002**: `/chat` endpoint returns responses sourced from the textbook (not hallucinations).
- **SC-003**: Zero "OpenAI API Key" errors during normal operation.
