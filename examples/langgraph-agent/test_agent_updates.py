"""Offline tests for dated updates; no database or provider calls."""
import importlib.util
import json
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage

os.environ['SEMAPHORE_LIMIT'] = '2'
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
helper = types.ModuleType('quickstart_deepseek_ollama_neo4j')
helper.create_graphiti = AsyncMock()
sys.modules[helper.__name__] = helper
spec = importlib.util.spec_from_file_location('updates_test_target',
                                             Path(__file__).with_name('agent_updates.py'))
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)
SOURCE = Path(__file__).with_name('history_updates.json')


class UpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_and_ground_truth_are_separate(self):
        messages, digest = agent.load_history(str(SOURCE))
        self.assertEqual(len(messages), 42)
        self.assertEqual(len(digest), 64)
        self.assertTrue(messages[0]['content'].startswith('[2026-10-01'))
        truth = json.loads(SOURCE.with_name('history_updates_expected.json').read_text())
        self.assertEqual(truth['user_turns'], 21)
        self.assertEqual(truth['expected_final']['budget_usd'], 90)
        self.assertNotIn('snapshots', json.loads(SOURCE.read_text()))

    async def test_prepare_ingests_separate_dated_episodes_and_can_resume(self):
        graphiti = SimpleNamespace(driver=None,
                                  add_episode=AsyncMock(return_value=SimpleNamespace(edges=[])))
        with (
            patch.object(agent.EpisodicNode, 'get_by_group_ids', new=AsyncMock(return_value=[])),
            patch.object(agent.EntityNode, 'get_by_group_ids', new=AsyncMock(return_value=[])),
        ):
            await agent.prepare_updates(graphiti, 'group', str(SOURCE))
        calls = graphiti.add_episode.await_args_list
        self.assertEqual(len(calls), 21)
        times = [call.kwargs['reference_time'] for call in calls]
        self.assertEqual(len(set(times)), 21)
        self.assertEqual(times, sorted(times))
        self.assertTrue(calls[-1].kwargs['name'].startswith('Experiment history '))
        # Earlier episodes cannot contain subsequent updates.
        self.assertNotIn('maximum budget to 90', calls[0].kwargs['episode_body'])
        self.assertEqual(calls[0].kwargs['group_id'], 'group')
        graphiti.add_episode.reset_mock()
        existing = [SimpleNamespace(name=call.kwargs['name']) for call in calls]
        with (
            patch.object(agent.EpisodicNode, 'get_by_group_ids',
                         new=AsyncMock(return_value=existing)),
            patch.object(agent.EntityNode, 'get_by_group_ids', new=AsyncMock(return_value=[])),
        ):
            await agent.prepare_updates(graphiti, 'group', str(SOURCE))
        graphiti.add_episode.assert_not_awaited()

    async def test_timestamp_order_is_validated(self):
        source = json.loads(SOURCE.read_text())
        source['messages'][1]['timestamp'] = source['messages'][0]['timestamp']
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                agent.load_history(str(path))

    async def test_recall_excludes_expired_invalid_and_future_facts(self):
        now = datetime.now(timezone.utc)
        def edge(uuid, **kwargs):
            return SimpleNamespace(uuid=uuid, group_id='g', expired_at=None,
                                   valid_at=kwargs.get('valid_at'),
                                   invalid_at=kwargs.get('invalid_at'))
        edges = [edge('current'), edge('old', invalid_at=now - timedelta(days=1)),
                 edge('future', valid_at=now + timedelta(days=1))]
        graphiti = SimpleNamespace(driver=None, search=AsyncMock(return_value=[]))
        with patch.object(agent.EntityEdge, 'get_by_node_uuid',
                          new=AsyncMock(return_value=edges)):
            await agent.recall_customer_facts(graphiti, {
                'messages': [HumanMessage(content='My current needs?')],
                'user_name': 'jess', 'user_node_uuid': 'j',
            }, 'g', True)
        ids = graphiti.search.await_args.kwargs['search_filter'].edge_uuids
        self.assertEqual(ids, ['current'])


if __name__ == '__main__':
    unittest.main()
