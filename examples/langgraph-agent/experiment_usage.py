"""Provider usage accounting, including Graphiti ingestion and format retries."""
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def read_records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def normalize_usage(usage):
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    usage = usage or {}
    input_tokens = usage.get('prompt_tokens', usage.get('input_tokens'))
    output_tokens = usage.get('completion_tokens', usage.get('output_tokens'))
    details = usage.get('prompt_tokens_details') or usage.get('input_token_details') or {}
    hit = usage.get('prompt_cache_hit_tokens', details.get('cached_tokens',
                                                       details.get('cache_read')))
    miss = usage.get('prompt_cache_miss_tokens')
    if miss is None and hit is not None and input_tokens is not None:
        miss = input_tokens - hit
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'cache_hit_tokens': hit, 'cache_miss_tokens': miss}


class PreparationUsage:
    def __init__(self, path, group_id):
        self.path = str(Path(path).resolve())
        self.group_id = group_id
        self.session_id = uuid4().hex
        self.phase = 'setup'
        self.item = None
        self.resource = None
        self.original = None
        self.phase_started = perf_counter()
        self.record('session_start')

    def record(self, event, **kwargs):
        append_record(self.path, {
            'event': event, 'group_id': self.group_id, 'session_id': self.session_id,
            'phase': self.phase, 'item': self.item,
            'timestamp': datetime.now(timezone.utc).isoformat(), **kwargs,
        })

    def begin_phase(self, phase):
        self.phase = phase
        self.item = None
        self.phase_started = perf_counter()
        self.record('phase_start')

    def finish_phase(self):
        self.record('phase_complete', seconds=perf_counter() - self.phase_started)

    def attach(self, graphiti):
        # Wrap the provider call, below Graphiti parsing/validation/retry layers.
        self.resource = graphiti.llm_client.client.chat.completions
        self.original = self.resource.create

        async def measured_create(*args, **kwargs):
            request_id = uuid4().hex
            phase, item = self.phase, self.item
            self.record('api_start', request_id=request_id, model=kwargs.get('model'))
            started = perf_counter()
            try:
                response = await self.original(*args, **kwargs)
            except Exception as exc:
                self.record('api_result', request_id=request_id, phase=phase, item=item,
                            status='error', error_type=type(exc).__name__,
                            seconds=perf_counter() - started,
                            **normalize_usage(None))
                raise
            self.record('api_result', request_id=request_id, phase=phase, item=item,
                        status='ok', model=getattr(response, 'model', kwargs.get('model')),
                        seconds=perf_counter() - started,
                        **normalize_usage(getattr(response, 'usage', None)))
            return response

        self.resource.create = measured_create

    def detach(self):
        if self.resource is not None:
            self.resource.create = self.original


def group_preparation(path, group_id):
    records = [row for row in read_records(path) if row.get('group_id') == group_id]
    finished = {row['phase'] for row in records if row['event'] == 'phase_complete'}
    if not {'catalog', 'user_creation', 'customer_updates'}.issubset(finished):
        raise ValueError('No complete costed preparation for this group. Prepare a NEW group '
                         'with the new script and --prep-usage-file; '
                         'old costs cannot be recovered.')
    return records


def usage_total(rows):
    keys = ('input_tokens', 'output_tokens', 'cache_hit_tokens', 'cache_miss_tokens')
    totals = {'calls': len(rows)}
    for key in keys:
        values = [row.get(key) for row in rows]
        totals[key] = sum(values) if all(v is not None for v in values) else None
        totals[f'known_{key}'] = sum(v for v in values if v is not None)
    totals['usage_complete'] = all(
        row.get('status', 'ok') == 'ok' and row.get('input_tokens') is not None
        and row.get('output_tokens') is not None for row in rows
    )
    return totals


def price_total(usage, pricing):
    if not pricing:
        return None
    keys = ('cache_hit_tokens', 'cache_miss_tokens', 'output_tokens')
    if not usage.get('usage_complete') or any(usage.get(key) is None for key in keys):
        return None
    return sum(usage[key] * pricing[f'{key}_per_million'] / 1_000_000 for key in keys)
