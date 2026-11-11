# Re-running the update comparison with preparation costs

The old experiment saved answering usage only. Its historical ingestion usage
cannot be recovered from graph edges. Use a NEW group and the updated script.
Do not overwrite or relabel the old experimental records.

## 1. Prepare once, recording every returned provider usage

From the repository root:

```powershell
.\.venv\Scripts\python.exe examples\langgraph-agent\agent_updates.py --prepare-history examples\langgraph-agent\history_updates.json --prep-usage-file results\costed_updates\preparation.jsonl
```

Record the group ID printed by the command. Five product episodes, generic user
creation and 21 dated conversation episodes are all metered. Twenty-one episodes
are NOT necessarily 21 API requests. Node extraction, edge extraction, resolution,
summaries and format retries can each generate requests. The wrapper records
provider usage BEFORE JSON parsing, so a paid malformed response still counts.
Each request start and result is written immediately to the ledger. No prompts,
API keys or response content are stored in that ledger.

If interrupted after the catalog/user stages complete, resume with the printed
group ID and the same `--prepare-history` and `--prep-usage-file`. Completed
updates are skipped; previously logged costs remain included. A killed request
with no returned usage is marked unresolved by the summary, not treated as free.

## 2. Run all six evaluations

```powershell
.\.venv\Scripts\python.exe examples\langgraph-agent\run_formal_updates.py --group 'REPLACE_WITH_NEW_GROUP' --prep-usage-file results\costed_updates\preparation.jsonl
```

This schedules each mode three times, in alternating order, using separate
processes. All evaluations are read-only. Each run records cache-hit input,
cache-miss input and output tokens when supplied by the provider. The runner
archives logs, per-run usage, source snapshots and preparation usage in a new
`results/formal_updates_TIMESTAMP` directory.

## 3. Read total_cost_summary.json

For each method the summary reports:

- Shared catalog and generic-user preparation (assigned once to each method
  when comparing deployments using the same product search).
- Customer-update ingestion (Graphiti only; full-history mode reads raw turns).
- All three answering runs.
- Method total and amortized input/output per query, including preparation.

The actual experiment total counts shared preparation only once, plus the
Graphiti memory preparation and all six evaluations. This differs from the sum
of hypothetical method totals, which each include their own shared setup.

One-time preparation is not multiplied by three and is not silently omitted.
Input and output tokens remain separate because prices can differ. This is API
usage accounting; local Ollama compute and database costs are not included.
If usage is absent or a request failed, complete totals may be null; known
usage is preserved separately. Provider-internal retries with no usage response
cannot be priced exactly from client logs: check the billing dashboard.

## Optional: convert tokens to money

Supply verified rates applicable to your account/model/date. No rate is guessed
or silently hardcoded. A JSON file should contain `currency`,
`cache_hit_tokens_per_million`, `cache_miss_tokens_per_million` and
`output_tokens_per_million`, all in the same currency. Include date/source fields
for provenance. Then run:

```powershell
.\.venv\Scripts\python.exe examples\langgraph-agent\summarize_update_costs.py --group 'REPLACE_WITH_NEW_GROUP' --prep-usage-file results\costed_updates\preparation.jsonl --results-file results\formal_updates_TIMESTAMP\runs.jsonl --output results\formal_updates_TIMESTAMP\total_cost_with_prices.json --pricing-file YOUR_VERIFIED_RATES.json
```

Without verified rates, `monetary_cost` is null; token totals are still reported.
Money is also null if required cache usage is missing, rather than assuming an
incorrect cache billing rate. Preparation and answering latency are separate;
the existing turn-latency metric is not renamed to total execution time.
