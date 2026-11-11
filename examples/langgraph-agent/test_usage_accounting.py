import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

from experiment_usage import PreparationUsage, group_preparation, price_total, read_records
from summarize_update_costs import summarize


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_money_uses_cache_and_output_rates_only_when_known(self):
        rates = {'cache_hit_tokens_per_million': 1, 'cache_miss_tokens_per_million': 2,
                 'output_tokens_per_million': 3, 'currency': 'test-currency'}
        usage = {'cache_hit_tokens': 40, 'cache_miss_tokens': 60,
                 'output_tokens': 20, 'usage_complete': True}
        self.assertAlmostEqual(price_total(usage, rates), 0.00022)
        self.assertIsNone(price_total({**usage, 'cache_hit_tokens': None}, rates))

    async def test_counts_every_returned_provider_response_and_failed_calls(self):
        response = SimpleNamespace(model='deepseek-flash', usage={
            'prompt_tokens': 100, 'completion_tokens': 20,
            'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60,
        })
        original = AsyncMock(side_effect=[response, response, RuntimeError('provider failed')])
        resource = SimpleNamespace(create=original)
        graphiti = SimpleNamespace(llm_client=SimpleNamespace(client=SimpleNamespace(
            chat=SimpleNamespace(completions=resource))))
        with TemporaryDirectory() as directory:
            ledger = str(Path(directory) / 'usage.jsonl')
            tracker = PreparationUsage(ledger, 'g')
            tracker.attach(graphiti)
            tracker.begin_phase('customer_updates')
            await resource.create(model='deepseek-flash')
            # Even if the caller rejects the first JSON response, both requests count.
            await resource.create(model='deepseek-flash')
            with self.assertRaises(RuntimeError):
                await resource.create(model='deepseek-flash')
            tracker.detach()
            calls = [row for row in read_records(ledger) if row['event'] == 'api_result']
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(row['input_tokens'] or 0 for row in calls), 200)
        self.assertEqual(calls[0]['cache_hit_tokens'], 40)
        self.assertIsNone(calls[2]['input_tokens'])
        self.assertEqual(calls[2]['status'], 'error')
        self.assertIs(resource.create, original)

    async def test_old_groups_without_preparation_usage_are_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'NEW group'):
                group_preparation(str(Path(directory) / 'missing.jsonl'), 'old-group')

    async def test_shared_setup_and_memory_costs_are_not_double_counted(self):
        def call(phase, tokens, output):
            return {'event': 'api_result', 'phase': phase, 'status': 'ok',
                    'input_tokens': tokens, 'output_tokens': output,
                    'cache_hit_tokens': 0, 'cache_miss_tokens': tokens,
                    'request_id': phase}
        prep = [call('catalog', 10, 2), call('user_creation', 5, 1),
                call('customer_updates', 100, 20)]
        starts = [{'event': 'api_start', 'phase': row['phase'],
                   'request_id': row['request_id']} for row in prep]
        runs = [{'memory_mode': mode, 'status': 'ok',
                 'call_metrics': [call('answer', tokens, 3)]}
                for mode, tokens in [('graphiti', 30), ('history', 40)]]
        result = summarize(prep + starts, runs)
        self.assertEqual(result['methods']['graphiti']['total_for_method']['input_tokens'], 145)
        self.assertEqual(result['methods']['history']['total_for_method']['input_tokens'], 55)
        self.assertEqual(result['actual_experiment_usage_counting_preparation_once']
                         ['input_tokens'], 185)
        self.assertIsNone(result['methods']['graphiti']['total_for_method']['monetary_cost'])
        prep.append({'event': 'api_start', 'request_id': 'unresolved',
                     'phase': 'customer_updates'})
        incomplete = summarize(prep + starts, runs)
        self.assertFalse(incomplete['methods']['graphiti']['total_for_method']['usage_complete'])
        self.assertIsNone(incomplete['methods']['graphiti']['total_for_method']['input_tokens'])
        self.assertTrue(incomplete['methods']['history']['total_for_method']['usage_complete'])


if __name__ == '__main__':
    unittest.main()
