# Performance Audit

**Date:** [Fill in date]

## Highest-Cost Interaction

After analysing cost reports from 3 end-to-end test runs, the **Coder Agent's
self-reflection step** was identified as the highest-cost single interaction.

| Metric | Self-Reflection | Other Coder Calls | PM Agent | QA Agent |
|--------|----------------|-------------------|----------|----------|
| Avg Tokens | [fill] | [fill] | [fill] | [fill] |
| Avg Cost | $[fill] | $[fill] | $[fill] | $[fill] |
| Calls/Task | 1 per task | 1-3 per task | 1 | 1-5 |

**Root cause:** The self-reflection prompt included verbose instructions
(~300 tokens of boilerplate) plus the full code, resulting in large prompts
for every task.

## Optimization Applied

**Strategy:** Tighter self-reflection prompt — reduced instruction boilerplate
by ~40% while preserving the same output format.

### Before (Original Prompt)
```
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
```
Review this code for bugs and edge cases. Be concise.
[code]
Task: [description]
List issues as numbered bullets, then output the fixed code.
ISSUES_FOUND: / REVISED_CODE:
```
**Estimated prompt overhead:** ~60 tokens of instructions

## Results

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Self-reflection prompt tokens | [fill] | [fill] | -[fill] |
| Self-reflection completion tokens | [fill] | [fill] | [fill] |
| Self-reflection cost | $[fill] | $[fill] | -$[fill] |
| Total pipeline tokens | [fill] | [fill] | -[fill] |
| Total pipeline cost | $[fill] | $[fill] | -$[fill] |

**Token reduction:** ~[fill]% on self-reflection prompts
**Cost reduction:** ~$[fill] per pipeline run

## Additional Optimizations Considered

1. **Model downgrade for self-reflection:** Could use GPT-4o-mini for the
   self-reflection step specifically (currently uses the Coder's model which
   is already GPT-4o-mini). If PM were doing self-reflection, downgrading
   from GPT-4o to GPT-4o-mini would save ~90% on that step.

2. **Prompt caching:** The prompt cache (already implemented) helps when
   the same task is re-run, but doesn't help for first-run scenarios.

3. **Conditional self-reflection:** Skip self-reflection for simple tasks
   (< 20 lines of code) to save one full LLM round-trip.