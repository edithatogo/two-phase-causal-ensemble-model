# Agent Instructions for chronus_causus Development

This file provides guidance for AI agents working on the `chronus_causus` library.

## Key Development Files & Procedures:

1.  **`roadmap.md`:**
    *   This is the primary guide for development.
    *   Refer to it for the overall plan and specific implementation phases.
    *   **Action:** After completing a significant step or phase from the roadmap, update `roadmap.md` to reflect this progress (e.g., mark items as "In Progress", "Completed", or add notes).

2.  **`todo.md`:**
    *   This file tracks the immediate next steps and a checklist of major features/phases.
    *   **Action:**
        *   Consult `todo.md` at the beginning of a work session to identify current priorities.
        *   Update `todo.md` by checking off completed items and adding new specific sub-tasks as they are identified.
        *   Ensure the "Next Immediate Action" section is current.

3.  **`sessionlog.md`:**
    *   This file logs interactions, decisions, and actions taken during development sessions.
    *   **Action:** Append to this log during each work session, noting:
        *   User requests.
        *   Agent's understanding or interpretation.
        *   Significant actions taken (e.g., file creations, code modifications, tool executions).
        *   Outcomes of actions (e.g., test results, errors encountered).

4.  **`AGENTS.md` (This file):**
    *   If new standing instructions or conventions are established for agent work on this library, update this file.

## General Workflow:

*   Follow the steps outlined in `roadmap.md`.
*   Use `todo.md` for granular task management.
*   Document your work in `sessionlog.md`.
*   When pushing changes:
    *   Ensure `roadmap.md`, `todo.md`, `sessionlog.md`, and this `AGENTS.md` file are up-to-date with the work included in the commit.
    *   Write clear and descriptive commit messages.

## Coding Conventions:

*   Follow PEP 8 for Python code.
*   Aim for a scikit-learn-like API design as specified in the roadmap.
*   Write unit tests for new functionality.

## Current Focus (as of initialization):

*   Proceeding with Phase 0 of `roadmap.md`.
*   Maintaining the integrity and usefulness of the tracking files listed above.
