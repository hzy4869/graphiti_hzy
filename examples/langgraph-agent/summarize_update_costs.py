"""Combine preparation and answering usage without double-counting shared setup."""
import argparse
import json
from pathlib import Path

from experiment_usage import group_preparation, price_total, read_records, usage_total


def summarize(preparation, runs, pricing=None):
    calls = [row for row in preparation if row['event'] == 'api_result']
    starts = {row['request_id'] for row in preparation if row['event'] == 'api_start'}
    completed = {row['request_id'] for row in calls}
    unknown = starts - completed
    common = [row for row in calls if row['phase'] in ('catalog', 'user_creation')]
    memory = [row for row in calls if row['phase'] == 'customer_updates']
    totals = {}
    for mode in ('graphiti', 'history'):
        selected = [row for row in runs if row['memory_mode'] == mode]
        evaluation = [
            {**call, 'status': row['status']} for row in selected
            for call in row['call_metrics']
        ]
        errors = sum(row['status'] != 'ok' for row in selected)
        phases = {
            'shared_catalog_and_user_setup': usage_total(common),
            'customer_memory_preparation': usage_total(memory if mode == 'graphiti' else []),
            'answering_all_runs': usage_total(evaluation),
        }
        total = usage_total(common + (memory if mode == 'graphiti' else []) + evaluation)
        relevant_unknown = [row for row in preparation if row['event'] == 'api_start'
                            and row['request_id'] in unknown
                            and (mode == 'graphiti' or row['phase'] != 'customer_updates')]
        if relevant_unknown or errors:
            total['usage_complete'] = False
            total['input_tokens'] = total['output_tokens'] = None
        for label, usage in phases.items():
            phase_names = {
                'shared_catalog_and_user_setup': ('catalog', 'user_creation'),
                'customer_memory_preparation': ('customer_updates',)
                if mode == 'graphiti' else (),
                'answering_all_runs': (),
            }[label]
            if any(row['phase'] in phase_names for row in relevant_unknown):
                usage['usage_complete'] = False
                usage['input_tokens'] = usage['output_tokens'] = None
        for usage in [*phases.values(), total]:
            usage['monetary_cost'] = price_total(usage, pricing)
        totals[mode] = {
            'evaluation_runs': len(selected), 'failed_evaluations': errors,
            'unresolved_preparation_requests': len(relevant_unknown),
            'phases': phases, 'total_for_method': total,
            'amortized_input_tokens_per_query': (
                total['input_tokens'] / len(selected)
                if selected and total['usage_complete'] else None
            ),
            'amortized_output_tokens_per_query': (
                total['output_tokens'] / len(selected)
                if selected and total['usage_complete'] else None
            ),
        }
    evaluation_all = [
        {**call, 'status': row['status']} for row in runs for call in row['call_metrics']
    ]
    actual = usage_total(calls + evaluation_all)
    if unknown or any(row['status'] != 'ok' for row in runs):
        actual['usage_complete'] = False
        actual['input_tokens'] = actual['output_tokens'] = None
    actual['monetary_cost'] = price_total(actual, pricing)
    return {
        'currency': pricing.get('currency') if pricing else None,
        'pricing': pricing, 'methods': totals,
        'actual_experiment_usage_counting_preparation_once': actual,
        'notes': [
            'Shared product graph and generic user creation are charged to BOTH methods '
            'for method comparison, but only ONCE in actual experiment usage.',
            'Only Graphiti is charged customer-update extraction. Full-history mode '
            'reads raw messages directly and performs no such extraction.',
            'Method totals include preparation once plus all selected evaluation runs; '
            'amortized values divide this sum by the selected query count.',
            'Missing/failed/in-flight usage is not assumed free. Provider-internal '
            'network retries without returned usage cannot be reconstructed; '
            'reconcile exact currency charges with the provider billing statement.',
            'Monetary cost is null unless cache usage and explicit price rates are available. '
            'Local embedding compute, Neo4j fees and electricity are outside API token usage.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep-usage-file', required=True)
    parser.add_argument('--results-file', required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pricing-file', help='Optional user-verified per-million rates JSON.')
    args = parser.parse_args()
    prep = group_preparation(args.prep_usage_file, args.group)
    runs = [row for row in read_records(args.results_file) if row['group_id'] == args.group]
    if not runs or {row['memory_mode'] for row in runs} != {'graphiti', 'history'}:
        parser.error('Need results for both graphiti and history in the selected group.')
    if len({row['history_sha256'] for row in runs}) != 1 or len({
        row['catalog_scope_sha256'] for row in runs
    }) != 1:
        parser.error('History or catalog fingerprints differ across evaluations.')
    pricing = None
    if args.pricing_file:
        pricing = json.loads(Path(args.pricing_file).read_text(encoding='utf-8-sig'))
        for key in ('cache_hit_tokens_per_million', 'cache_miss_tokens_per_million',
                    'output_tokens_per_million'):
            if not isinstance(pricing.get(key), (int, float)) or pricing[key] < 0:
                parser.error(f'Pricing must supply a nonnegative rate for {key}.')
        if not pricing.get('currency'):
            parser.error('Pricing must specify currency.')
    report = summarize(prep, runs, pricing)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for mode, result in report['methods'].items():
        print(mode, json.dumps(result['total_for_method'], ensure_ascii=False))
    print(f'Saved total API usage comparison to {target}')


if __name__ == '__main__':
    main()
