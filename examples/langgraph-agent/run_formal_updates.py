"""Run six read-only evaluation subprocesses; preserve every scheduled outcome."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from experiment_usage import group_preparation
from summarize_update_costs import summarize


REPO = Path(__file__).resolve().parents[2]
QUESTION = (
    'State my current shoe size, maximum budget, required color, and material restrictions, '
    'then recommend shoes for these needs. Use my latest requirements and say when '
    'no match can be verified.'
)
PLAN = [(1, 'graphiti'), (1, 'history'), (2, 'history'),
        (2, 'graphiti'), (3, 'graphiti'), (3, 'history')]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, help='New group with measured preparation.')
    parser.add_argument('--prep-usage-file', default='results/update_preparation_usage.jsonl')
    args = parser.parse_args()
    group = args.group
    ledger = str(Path(args.prep_usage_file).resolve())
    preparation = group_preparation(ledger, group)
    folder = REPO / 'examples' / 'langgraph-agent'
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = REPO / 'results' / f'formal_updates_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    sources = {}
    for name in ('agent_updates.py', 'experiment_usage.py', 'summarize_update_costs.py',
                 'run_formal_updates.py', 'history_updates.json', 'history_updates_expected.json'):
        source = folder / name
        shutil.copy2(source, snapshots / name)
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / 'preparation_usage.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in preparation), encoding='utf-8'
    )
    versions = {}
    for package in ('graphiti-core', 'langgraph', 'langchain-openai', 'openai', 'neo4j'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'group_id': group, 'question': QUESTION, 'python': sys.version,
        'preparation_usage_file': ledger,
        'package_versions': versions, 'source_sha256': sources,
        'plan': [{'trial': trial, 'mode': mode} for trial, mode in PLAN],
        'read_only': True, 'automatic_retries_of_runs': False,
        'note': 'Each trial is a fresh process on the same prepared graph. '
                'Provider SDK retries may occur inside a turn. '
                'Turn metrics exclude setup. Preparation usage is archived separately '
                'and included in total_cost_summary.json.',
    }
    save(output / 'manifest.json', manifest)
    print(f'OUTPUT_DIR={output}', flush=True)
    records = []
    for order, (trial, mode) in enumerate(PLAN, 1):
        name = f'{order:02d}_trial{trial}_{mode}'
        result_path = output / f'{name}.jsonl'
        command = [
            sys.executable, '-X', 'utf8', str(folder / 'agent_updates.py'),
            '--resume-group', group, '--memory-mode', mode,
            '--prep-usage-file', ledger,
            '--history-file', str(folder / 'history_updates.json'),
            '--question', QUESTION, '--results-file', str(result_path),
        ]
        print(f'START {order}/6 trial={trial} mode={mode}', flush=True)
        exit_code = None
        error = None
        try:
            with (output / f'{name}.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=REPO, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=300)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            error = 'Process exceeded 300 seconds; no automatic rerun.'
        row = {'order': order, 'trial': trial, 'mode': mode,
               'exit_code': exit_code, 'process_error': error}
        if result_path.exists():
            lines = result_path.read_text(encoding='utf-8').splitlines()
            if len(lines) == 1:
                row.update(json.loads(lines[0]))
            else:
                row['status'] = 'invalid_result_count'
        else:
            row['status'] = 'no_result'
        records.append(row)
        with (output / 'runs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'END {order}/6 status={row.get("status")} exit={exit_code} '
              f'seconds={row.get("turn_seconds")} input={row.get("input_tokens")}', flush=True)
    summary = {}
    for mode in ('graphiti', 'history'):
        successes = [row for row in records if row['mode'] == mode
                     and row.get('status') == 'ok' and row['exit_code'] == 0]
        stats = {'scheduled': 3, 'successful': len(successes)}
        for key in ('turn_seconds', 'input_tokens', 'output_tokens', 'tool_calls'):
            values = [row[key] for row in successes if row.get(key) is not None]
            stats[key] = {'n': len(values),
                          'mean': statistics.mean(values) if values else None,
                          'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
        summary[mode] = stats
    summary['same_history_fingerprint'] = len({r.get('history_sha256') for r in records}) == 1
    summary['same_catalog_scope'] = len({r.get('catalog_scope_sha256') for r in records}) == 1
    save(output / 'summary.json', summary)
    if all(row.get('status') == 'ok' and row['exit_code'] == 0 for row in records):
        save(output / 'total_cost_summary.json', summarize(preparation, records))
    else:
        print('Some evaluations failed. Preserve their logs; total cost summary is incomplete.',
              flush=True)
    fields = ['order', 'trial', 'mode', 'status', 'exit_code', 'turn_seconds',
              'input_tokens', 'output_tokens', 'tool_calls', 'answer']
    with (output / 'runs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
