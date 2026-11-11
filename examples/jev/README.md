# Jev dedupe with FalkorDB

## DeepSeek + Ollama + Neo4j comparison in this fork

`modified.py` runs the same four-episode timeline using the provider setup from
`examples/quickstart/quickstart_deepseek_ollama_neo4j.py`. It uses DeepSeek for
extraction, entity deduplication, and fact resolution; local Ollama `bge-m3`
for embeddings; and the Neo4j connection in the repository-root `.env`.
No Jev or OpenAI API key is needed. Start Ollama and run from the repository root:

```powershell
.\.venv\Scripts\python.exe examples\jev\modified.py
```

The script prints the extracted entities and facts after each episode, then
lists all facts with `valid_at`, `invalid_at`, and `expired_at`. Each run gets a
new `group_id` and leaves existing graph data intact. DeepSeek's extraction
and contradiction decisions can differ from the Jev example's output.

The requirements and commands below apply to the original Jev/FalkorDB script.

This example routes node deduplication and edge resolution to Jev. OpenAI
`gpt-6-luna` handles extraction, summaries, and all other prompts. The example
adds four episodes in order. The episodes show a name variant, a repeated
fact, and a transfer of responsibility for the customer billing platform from
Alice Chen to Priya Patel.

## Prerequisites

- Python 3.11+
- FalkorDB
- An OpenAI API key in `OPENAI_API_KEY`
- A TypeSafe API key in `JEV_API_KEY`

Start FalkorDB with Docker:

```bash
docker run -p 6379:6379 falkordb/falkordb:latest
```

Copy `.env.example` to `.env` and set both API keys. The example uses
`localhost:6379` by default. Set `FALKORDB_HOST`, `FALKORDB_PORT`,
`FALKORDB_USERNAME`, or `FALKORDB_PASSWORD` to use other connection settings.

## When to use a decision model

Jev is a decision model. It answers typed questions with probabilities. It does
not generate text. Use a general LLM as the default model. Route only
`dedupe_nodes.nodes` and `dedupe_edges.resolve_edge` to Jev. Do not use Jev for
extraction, summaries, or other generation tasks.

## Prompt routing

| Prompt | Model |
|---|---|
| Extraction, summaries, and all other prompts | OpenAI `gpt-6-luna` |
| `dedupe_nodes.nodes` | Jev `jev-latest` |
| `dedupe_edges.resolve_edge` | Jev `jev-latest` |

The example uses the default OpenAI embedder. TypeSafe hosts the Jev API.
Graphiti sends dedupe prompt content to TypeSafe.

## Expected output

One run showed Alice's billing-platform fact with
`invalid_at=2025-01-10T00:00:00+00:00` and an `expired_at` value. Priya's
replacement fact was active. The contradiction probability was `0.89` for
Alice's fact and `0.04` for the Acme fact. The run made 2
`dedupe_nodes.nodes` calls and 3 `dedupe_edges.resolve_edge` calls. The final
node list showed one Alice Chen entity.

Extraction output can change between runs. If the model does not extract
Alice's billing-platform fact, or if it writes the change as a separate
termination fact, Alice's fact stays active. The current Graphiti contradiction
rule uses only the new fact's `valid_at` value.

## Run the example

From the repository root, install the project dependencies:

```bash
uv sync --extra dev
```

Then run the example from this directory:

```bash
cd examples/jev
uv run python jev_dedupe_falkordb.py
```

The output lists the entities and facts for each episode. It then lists all
facts with their invalidation or expiration times and the final entity nodes.
The example does not call `Graphiti.search`.
