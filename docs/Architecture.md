## 1 Architecture Overview

```mermaid
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION GRAPH (Phase 3)                        │
│                                                                              │
│  ┌──────┐  SharedState  ┌──────────┐  SharedState  ┌───────────────────┐    │
│  │      │ ────────────► │          │ ────────────► │                   │    │
│  │  PM  │               │  CODER   │◄─────────────│   REVIEW LOOP     │    │
│  │      │               │          │  A2A Message  │                   │    │
│  └──────┘               └────┬─────┘               │  ┌─────┐  A2A    │    │
│                              │                      │  │ QA  │◄──────►│    │
│                              │ Self-Reflection      │  │Agent│ Message│    │
│                              │ (internal loop)      │  └─────┘        │    │
│                              ▼                      │  Max 5 iters    │    │
│                         ┌─────────┐                 └───────────────────┘    │
│                         │Critique │                                          │
│                         │& Revise │                                          │
│                         └─────────┘                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2 Data Flow:
1. PM → SharedState (tech spec + tasks)
2. Coder reads tasks → self-reflects → writes code → SharedState
3. Review Loop:
    a. Coder sends code to QA via A2A message (intent: "review_request")
    b. QA writes pytest tests, executes them, parses failures
    c. QA sends fix instructions via A2A message (intent: "fix_instructions")
    d. Coder revises code based on fix instructions
    e. Repeat until all tests pass OR 5 iterations reached
4. QA produces final report → SharedState