## 1 Architecture Overview

```mermaid
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION GRAPH                         │
│                                                                 │
│  ┌──────────┐    SharedState     ┌──────────┐    SharedState    │
│  │          │  ──────────────►   │          │  ──────────────►  │
│  │    PM    │   (tech_spec +     │  CODER   │   (code_output +  │
│  │  Agent   │    task_list)      │  Agent   │    task_status)   │
│  │          │                    │          │                    │
│  └──────────┘                    └──────────┘                    │
│       ▲                               │                         │
│       │         User Requirement      │      Final Output       │
│       └───────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## 2 Data Flow:
1- User submits a free-text requirement
2- PM Agent parses it → produces tech spec + task list → writes to SharedState
3- Handoff protocol: Coder reads tasks with status pending from SharedState
4- Coder Agent executes each task in order → updates status + code in SharedState
5- Orchestrator returns the final SharedState as structured output