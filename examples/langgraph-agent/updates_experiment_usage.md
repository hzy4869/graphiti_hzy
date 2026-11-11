# Repeated requirement updates

`agent_updates.py` is a separate copy of `agent_modified.py`. The original
script is preserved. `history_updates.json` is a synthetic dated conversation:
one initial profile, then 20 individual changes (five each to size, budget,
color and the material restriction list), with 42 total messages.

Unlike the short-history experiment, preparation ingests 21 sequential episodes
with distinct reference times. The final episode marks completion. Existing
episode names allow resuming partial preparation with the same group. Both
evaluation modes receive the same dated source; the history group receives the
entire transcript without truncation. Current Graphiti retrieval excludes facts
whose stored validity interval has ended, but this only works if Graphiti
correctly resolves the update. Old requirements are not deleted manually.

## Prepare once

From the repository root:

```powershell
.\.venv\Scripts\python.exe examples\langgraph-agent\agent_updates.py --prepare-history examples\langgraph-agent\history_updates.json
```

This imports five catalog products, creates the user and processes 21 conversation
episodes. Graphiti may make multiple DeepSeek calls per episode, so preparation
takes considerably more time and API usage than the two-message example. It is
not necessary to repeat preparation for every evaluation. If interrupted, use
the printed group with `--resume-group GROUP_ID --prepare-history ...`.

## Compare both modes

```powershell
$experimentGroup = 'REPLACE_WITH_NEW_UPDATES_GROUP'
$question = 'State my current shoe size, maximum budget, required color, and material restrictions, then recommend shoes for these needs. Use my latest requirements and say when no match can be verified.'

.\.venv\Scripts\python.exe examples\langgraph-agent\agent_updates.py --resume-group $experimentGroup --memory-mode graphiti --history-file examples\langgraph-agent\history_updates.json --question $question --results-file results\update_runs.jsonl

.\.venv\Scripts\python.exe examples\langgraph-agent\agent_updates.py --resume-group $experimentGroup --memory-mode history --history-file examples\langgraph-agent\history_updates.json --question $question --results-file results\update_runs.jsonl
```

Both evaluation commands create fresh threads and disable writing. Use a new
group for this case, not a group from the previous short-history experiment.
Repeat each mode three times; alternate execution order. Record errors too.

## Score the outcome

`history_updates_expected.json` contains final ground truth and all 21 state
snapshots for inspection. It is never loaded into the agent prompt. Final values:
US men's 9, maximum 90 USD, black, avoid wool. Adult shoes in the source catalog
cost 98/100/120 USD, so no known adult shoe satisfies this budget. A correct
no-match or explicitly unverified answer is preferable to an invented match.

Measure four current-value answers independently (0-4), stale-value errors,
unsupported price/material claims, tokens and latency. Observe preparation's
printed valid_at/invalid_at/expired_at fields to assess whether old facts were
actually invalidated. The JSONL token totals cover evaluation model calls only;
preparation overhead is separate. Larger history can increase potential token
savings but does not guarantee better answers or lower latency. Full-history
reasoning may also succeed: that is a valid experimental result.

The transcript is constructed, not a recording of live model conversations.
State this in the experiment description. Offline tests validate code paths;
actual model extraction and temporal conflict resolution still require running
the preparation and evaluation commands.
