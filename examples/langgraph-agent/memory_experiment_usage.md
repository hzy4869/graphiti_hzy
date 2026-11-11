# Comparing Graphiti facts with original conversation history

Run commands from the graphiti_hzy repository root. This comparison holds the
model, catalog-only product search, system instructions and current question
fixed. The changed variable is how prior conversation reaches the model.

## 1. Prepare the source conversation once

Edit `examples/langgraph-agent/history_example.json` for the case. Messages must
alternate user/assistant, starting with user. System and tool messages are rejected.
Then create a fresh group:

```powershell
.\.venv\Scripts\python.exe examples\langgraph-agent\agent_modified.py --prepare-history examples\langgraph-agent\history_example.json
```

This imports the catalog and ingests the supplied transcript through Graphiti.
It does not generate a recommendation or fabricate a model response. Copy the
printed group ID. Check extraction quality separately; incomplete extraction is
an experimental outcome, not a reason to silently replace facts.

## 2. Evaluate the same question in separate processes

```powershell
$experimentGroup = 'REPLACE_WITH_PREPARED_GROUP_ID'

.\.venv\Scripts\python.exe examples\langgraph-agent\agent_modified.py --resume-group $experimentGroup --memory-mode graphiti --history-file examples\langgraph-agent\history_example.json --question 'Recommend shoes based on my needs.' --results-file results\memory_runs.jsonl

.\.venv\Scripts\python.exe examples\langgraph-agent\agent_modified.py --resume-group $experimentGroup --memory-mode history --history-file examples\langgraph-agent\history_example.json --question 'Recommend shoes based on my needs.' --results-file results\memory_runs.jsonl
```

Both modes automatically use one fresh thread and disable conversation writing.
Graphiti mode injects retrieved customer facts, without the original transcript.
History mode loads the complete transcript as ordinary user/assistant messages,
without retrieving customer facts. The transcript fingerprint must match a
history preparation episode in the selected group. Product search excludes
conversation-derived edges in both modes. No catalog fallback or seeded facts
are permitted in this evaluation mode.

Use new groups for different cases, and keep the preparation file unchanged for
all repetitions of that case. Repeat each mode three times, alternating which
mode runs first to reduce ordering/cache effects. The same source information is
available to the two memory mechanisms; their token counts need not be equal.
Existing groups prepared by ad-hoc chat commands do not contain the required
fingerprint: prepare a new group from the fixed transcript for this comparison.

## 3. Interpret the JSONL records

Each line is a run, including question, final answer, group, history fingerprint,
catalog scope fingerprint, memory mode, model, status, tool trace and metrics.
Failed turns are recorded and the process still raises the original error.

- `input_tokens` / `output_tokens`: provider-reported totals across every agent
  LLM call in the turn, including the call after tool execution. These are billed
  request volumes, not unique conversation length. Unknown usage is `null`.
- `turn_seconds`: wall time from submitting the turn to completion, including
  recall, product searches and agent calls. Startup, catalog preparation, graph
  index initialization and node discovery are excluded. Graphiti evaluation has
  retrieval latency; history evaluation has direct transcript input.
- `llm_seconds`: total time spent in successful agent model calls, including SDK
  retries. Failed/incomplete runs may have only partial usage data.
- These evaluation token totals do not include the one-time Graphiti preparation
  calls. Track preparation cost separately if reporting total system cost.
- This file does not assign recommendation correctness or constraint scores.
  Compare with the case's ground truth using a scoring rule fixed in advance;
  saying that composition/price is unknown differs from proving a constraint.

The sample is synthetic. Use real recorded turns or clearly identify synthetic
transcripts in the experiment. This change does not create a course report or
presentation.
