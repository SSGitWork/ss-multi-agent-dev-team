# Performance Audit

## Highest-Cost Interaction

After analysing cost reports from 3 end-to-end test runs and an additional
optimized run of the CSV Processor task, the **Coder Agent's self-reflection
step** remains the highest-cost single interaction pattern (when it runs),
but the **per-call cost** has dropped significantly with the concise prompt.

For the CSV Processor task:

- **Before optimization**
  - Self-reflection: 14,931 tokens, **$0.0055**
- **After optimization**
  - Self-reflection: 10,896 tokens, **$0.0038**

So self-reflection is still a non-trivial contributor, but the optimized prompt
has meaningfully reduced its share of the total pipeline cost.

### Comparative Metrics (CSV Processor)

| Metric                      | Before (orig CSV) | After (CSV v2) |
|-----------------------------|-------------------|----------------|
| Self-reflection tokens      | 14,931            | 10,896         |
| Self-reflection cost        | $0.0055           | $0.0038        |
| Total pipeline tokens       | 400,876           | 428,464        |
| Total pipeline cost         | $0.1021           | $0.1040        |
| Self-reflection % of cost   | ~5.4%             | ~3.7%          |

> Note: The v2 run has a slightly higher overall cost because the total
> pipeline tokens increased (more iterations / other agents), even though
> self-reflection itself became cheaper.

## Root Cause

The original self-reflection prompt included:

- Verbose, repeated instructions (~280–300 tokens of boilerplate)
- The full code plus the full task description on each run

This combination made the self-reflection **prompt tokens** large. Because
self-reflection runs at least once per task, this overhead scaled linearly
with the number of tasks and code size, making it a noticeable cost driver.

In the optimized version, the self-reflection
prompt was tightened:

- Instruction boilerplate reduced to ~60 tokens
- Instructions made more compact and direct
- Same overall behavior and output format preserved

This reduces the prompt-side overhead per self-reflection call while keeping
its usefulness.

## Optimization Applied

**Strategy:** Tighter self-reflection prompt — reduced instruction boilerplate
by ~60–75% while preserving the same output format.

### Before (Original Prompt)
```text
## SELF-REFLECTION TASK
Critically analyse this code for potential issues.

## THE CODE
[full code]

## ORIGINAL TASK
[full task description]

## INSTRUCTIONS
1. List ALL potential bugs, edge cases, and issues you can find.
2. For each issue, describe the fix.
3. Then output the REVISED code that addresses all issues.

## OUTPUT FORMAT
Respond with EXACTLY this format:
...
```
**Estimated prompt overhead:** ~280 tokens of instructions

### After (Optimized Prompt)
```text
Review this code for bugs and edge cases. Be concise.
[code]
Task: [description]
List issues as numbered bullets, then output the fixed code.

ISSUES_FOUND:
[bullet list]
REVISED_CODE:
[full updated code]
```
**Estimated prompt overhead:** ~60 tokens of instructions

## Results

### Self-Reflection Step (CSV Processor)

| Metric                         | Before (orig CSV) | After (CSV v2) | Delta        |
|-------------------------------|-------------------|----------------|-------------|
| Self-reflection prompt tokens | ~7,600 (est.)     | ~6,000 (est.)  | ~ -21%      |
| Self-reflection completion    | 7,262             | 4,907          | -2,355      |
| Self-reflection total tokens  | 14,931            | 10,896         | -4,035      |
| Self-reflection cost          | $0.0055           | $0.0038        | **-$0.0017** |

### Whole Pipeline (CSV Processor)

| Metric                  | Before (orig CSV) | After (CSV v2) | Delta       |
|-------------------------|-------------------|----------------|------------|
| Total prompt tokens     | 329,790           | 361,112        | +31,322    |
| Total completion tokens | 71,086            | 67,352         | -3,734     |
| Total tokens            | 400,876           | 428,464        | +27,588    |
| Total pipeline cost     | $0.1021           | $0.1040        | +$0.0019   |

Interpretation:

- The **per-call cost** of self-reflection has dropped (fewer tokens, less $).
- The **overall pipeline cost** in the v2 CSV run is slightly higher because
  other parts of the process consumed more tokens (more iterations / QA / revision).
- The optimization is still valuable: if you hold everything else constant,
  the new prompt is cheaper whenever self-reflection runs.

**Token reduction (self-reflection step):** ~27% fewer tokens  
**Cost reduction (self-reflection step):** ~31% lower for that step

## Additional Optimizations Considered

1. **Model downgrade for self-reflection**  
   Self-reflection is already running on `azure/gpt-4o-mini` according to the
   cost reports, so there is no further downgrade opportunity at the moment.
   If a future version used a larger model for self-reflection (e.g., `gpt-4o`
   or `gpt-5.x`), switching back to `gpt-4o-mini` or similar would provide
   significant savings.

2. **Prompt caching**  
   The prompt cache (already implemented) helps when:
   - The same task and code are re-run with identical or very similar prompts.
   For first-run scenarios or when code changes substantially between runs,
   prompt caching provides limited benefit.

3. **Conditional self-reflection**  
   To reduce the number of self-reflection calls:
   - **Skip self-reflection for simple tasks** (e.g., code < 20–30 lines or
     “low risk” categories).
   - **Gate on QA results**: Only trigger self-reflection if QA fails, or if
     the PM marks the task as complex.
   - **Size-aware behavior**: For very large code artifacts, run a cheap
     static check first and call full self-reflection only when issues are
     detected.

4. **Scope-limited self-reflection**  
   When only a small portion of the code changes, restrict self-reflection to:
   - The modified sections
   - The functions impacted by a bug or new requirement  
   This reduces the amount of code that needs to be re-sent on each call.

5. **Shared instructions / system-level boilerplate**  
   Where possible, move stable, generic instructions into:
   - System messages
   - Shared configuration  
   And keep the per-call messages focused on the specific code and task context,
   further minimizing prompt overhead.