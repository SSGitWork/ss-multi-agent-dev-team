# Reflection Write-Up

## Three Most Significant Design Decisions

### 1. Shared State over Message Passing

The most consequential architectural decision was using a single typed
`SharedState` Pydantic model as the communication backbone between agents,
rather than a pure message-passing architecture.

**Why:** In a three-agent system where each agent builds on the previous
agent's output, a shared state model provides several advantages. First,
it gives every agent full visibility into the pipeline's current status —
the Coder can see the PM's spec, the QA can see the Coder's accumulated
code, and the orchestrator can inspect everything. Second, Pydantic
validation ensures that no agent can write malformed data — if the PM
produces an invalid task list, the error is caught immediately at the
schema boundary rather than propagating silently. Third, the entire state
is JSON-serializable, which makes debugging trivial: you can snapshot the
state at any point and replay from there.

The trade-off is tighter coupling — agents must agree on the SharedState
schema. In a larger system with dozens of agents, this would become
unwieldy and a message-passing architecture (like A2A everywhere) would
scale better. For three agents, the simplicity won.

### 2. A2A Protocol for Coder, QA (not MCP)

For the iterative review loop between Coder and QA, I chose the A2A
(Agent-to-Agent) protocol over MCP (Model Context Protocol).

**Why:** The Coder and QA are peers — they exchange work products (code,
test results, fix instructions) in a conversational back-and-forth. A2A's
intent-based routing (`REVIEW_REQUEST`, `FIX_INSTRUCTIONS`,
`ALL_TESTS_PASSED`) maps naturally to this interaction pattern. Each
message carries a `correlation_id` that links an entire review cycle,
making it easy to trace "which fix instruction was in response to which
code version." MCP, by contrast, models a client-server relationship
where one agent calls tools on another — that's the wrong abstraction
for peer review.

The A2A message bus also provides a complete audit trail. After a pipeline
run, you can inspect every message exchanged between Coder and QA, which
is invaluable for debugging why a particular fix didn't work.

### 3. Model Assignment by Role (GPT-4o for PM, GPT-4o-mini for Coder/QA)

I assigned the larger, more expensive GPT-4o model to the PM agent and
the smaller GPT-4o-mini to the Coder and QA agents.

**Why:** The PM's job is the most cognitively demanding — it must parse
ambiguous natural-language requirements, identify implicit constraints,
and produce a coherent task decomposition. This benefits from the stronger
reasoning capabilities of GPT-4o. The Coder and QA, by contrast, perform
more structured tasks (write code following a spec, write tests for given
criteria) where GPT-4o-mini performs nearly as well at a fraction of the
cost. In my testing, this split reduced per-pipeline cost by approximately
60% compared to using GPT-4o for all agents, with no measurable quality
degradation on the Coder/QA outputs.

---

## Hardest Bug

The hardest bug I encountered was the **ChromaDB Rust panic on Windows**
during Phase 1 setup. When initializing `chromadb.PersistentClient` with
ChromaDB 1.0.x, the application crashed with:

```
thread 'pm_agent' panicked at rust\sqlite\src\db.rs:157:42:
range start index 10 out of range for slice of length 9
```

This was a Rust-level panic in ChromaDB's new SQLite bindings — not a
Python exception — so there was no stack trace to debug, no try/except
that could catch it, and no error message that pointed to a fix. The
panic also corrupted the local database, so subsequent runs failed even
after code changes.

**Resolution:** I downgraded ChromaDB from 1.0.x to 0.5.x (the last
stable release before the Rust rewrite), deleted the corrupted
`chroma_store/` directory, and removed the `Settings` import that was
specific to the 1.0.x API. I also added a guard in the `retrieve()`
method to check `collection.count()` before querying, preventing a
separate edge case where querying an empty collection with `n_results > 0`
could cause errors.

This bug taught me to always pin dependencies to known-good versions in
`requirements.txt` rather than using latest, and to add graceful fallbacks
for infrastructure components (the memory system now degrades to
sliding-window-only if ChromaDB fails to initialize).

---

## What I Would Do Differently With More Time

**1. Streaming output and real-time feedback.** Currently, the pipeline
runs silently for minutes before producing output. With more time, I would
implement streaming — showing the PM's task list as it's generated, the
Coder's ReACT steps in real-time, and QA test results as they execute.
This would dramatically improve the user experience and make debugging
easier.

**2. Persistent conversation memory across sessions.** The current memory
system resets between pipeline runs. With more time, I would implement
session persistence — storing conversation history in ChromaDB with
session metadata so the agents can learn from past runs. For example, if
the Coder made a mistake on a similar task last week, the memory system
could surface that context and help avoid the same mistake.

**3. Parallel task execution.** Currently, the Coder processes tasks
sequentially. Many tasks are independent (e.g., "implement add function"
and "implement subtract function") and could be executed in parallel.
With more time, I would add dependency analysis to the PM's output and
implement concurrent task execution using Python's `asyncio` or
`concurrent.futures`, with proper SharedState locking.

**4. More sophisticated QA.** The current QA agent generates tests from
acceptance criteria, but it doesn't do static analysis, type checking,
or security scanning. With more time, I would integrate `mypy`, `ruff`,
and `bandit` as additional QA tools, giving the QA agent a multi-layered
review capability beyond just pytest.