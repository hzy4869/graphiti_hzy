"""Offline regression checks for the memory ablation; no API or database calls."""

import importlib.util
import json
import os
import sys
import types
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ['SEMAPHORE_LIMIT'] = '2'
os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
helper = types.ModuleType('quickstart_deepseek_ollama_neo4j')
helper.create_graphiti = AsyncMock()
sys.modules[helper.__name__] = helper
spec = importlib.util.spec_from_file_location('agent_memory_test_target',
                                             Path(__file__).with_name('agent_modified.py'))
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_validation_and_stable_fingerprint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'history.json'
            data = {'messages': [{'role': 'user', 'content': 'I prefer black shoes.'}]}
            path.write_text(json.dumps(data), encoding='utf-8')
            messages, digest = agent.load_history(str(path))
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            self.assertEqual(agent.load_history(str(path))[1], digest)
            self.assertEqual(messages, data['messages'])
            data['messages'][0]['role'] = 'system'
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                agent.load_history(str(path))

    async def test_history_input_and_metrics_include_all_model_calls(self):
        class FakeGraph:
            async def astream(self, state, config):
                self.state = state
                yield {'agent': {
                    'messages': [AIMessage(content='', tool_calls=[{
                        'name': 'get_shoe_data', 'args': {'query': 'shoes'},
                        'id': 'tool', 'type': 'tool_call',
                    }])],
                    'call_metrics': [{'input_tokens': 10, 'output_tokens': 2, 'seconds': 0.1}],
                }}
                yield {'agent': {
                    'messages': [AIMessage(content='Recommendation')],
                    'call_metrics': [{'input_tokens': 20, 'output_tokens': 3, 'seconds': 0.2}],
                }}

        graph = FakeGraph()
        prior = [{'role': 'user', 'content': 'I prefer black shoes.'}]
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'results.jsonl')
            result = await agent.run_turn(
                graph, {}, 'jess', 'Recommend shoes.', history=prior,
                metadata={'memory_mode': 'history'}, results_file=path,
            )
            saved = json.loads(Path(path).read_text(encoding='utf-8'))
        self.assertEqual(graph.state['messages'][:-1], prior)
        self.assertEqual(result['input_tokens'], 30)
        self.assertEqual(result['output_tokens'], 5)
        self.assertEqual(result['llm_calls'], 2)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(saved['answer'], 'Recommendation')
        self.assertEqual(saved['memory_mode'], 'history')

    async def test_failed_runs_are_recorded(self):
        class FailingGraph:
            async def astream(self, state, config):
                raise RuntimeError('test failure')
                yield

        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'error.jsonl')
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                await agent.run_turn(FailingGraph(), {}, 'jess', 'Question', results_file=path)
            saved = json.loads(Path(path).read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'error')
        self.assertIsNone(saved['input_tokens'])

    async def test_disabled_recall_does_not_query_user_facts(self):
        graphiti = SimpleNamespace(search=AsyncMock())
        with patch.object(agent.EntityEdge, 'get_by_node_uuid', new=AsyncMock()) as lookup:
            self.assertEqual(await agent.recall_customer_facts(graphiti, {}, 'g', False), [])
            lookup.assert_not_awaited()
            graphiti.search.assert_not_awaited()

    async def test_catalog_scope_excludes_conversation_and_mixed_sources(self):
        episodes = [
            SimpleNamespace(uuid='p', source_description='ManyBirds products'),
            SimpleNamespace(uuid='chat', source_description='Chatbot'),
        ]
        edges = [
            SimpleNamespace(uuid='catalog', episodes=['p']),
            SimpleNamespace(uuid='personal', episodes=['chat']),
            SimpleNamespace(uuid='mixed', episodes=['p', 'chat']),
            SimpleNamespace(uuid='unknown', episodes=[]),
        ]
        with (
            patch.object(agent.EpisodicNode, 'get_by_group_ids',
                         new=AsyncMock(return_value=episodes)),
            patch.object(agent.EntityEdge, 'get_by_group_ids', new=AsyncMock(return_value=edges)),
        ):
            actual = await agent.catalog_edge_uuids(SimpleNamespace(driver=None), 'g')
        self.assertEqual(actual, ['catalog'])

    async def test_both_modes_keep_product_tool_and_only_on_injects_memory(self):
        prompts, results = {}, {}
        for enabled in (True, False):
            captured = []

            class FakeModel:
                def __init__(self, **kwargs):
                    pass

                def bind_tools(self, tools, **kwargs):
                    return self

                async def ainvoke(self, messages):
                    captured.append(messages)
                    if isinstance(messages[-1], ToolMessage):
                        return AIMessage(
                            content='Available product information.',
                            usage_metadata={'input_tokens': 20, 'output_tokens': 3,
                                            'total_tokens': 23},
                        )
                    return AIMessage(content='', tool_calls=[{
                        'name': 'get_shoe_data', 'args': {'query': 'recommend shoes'},
                        'id': 'lookup', 'type': 'tool_call',
                    }])

            async def search(query, **kwargs):
                ids = kwargs['search_filter'].edge_uuids
                if ids == ['catalog']:
                    return [SimpleNamespace(fact='Catalog shoe sizes 8 through 14.')]
                self.assertTrue(enabled)
                self.assertEqual(ids, ['personal'])
                return [SimpleNamespace(fact='Jess prefers black shoes, budget 100.')]

            graphiti = SimpleNamespace(driver=None, search=AsyncMock(side_effect=search),
                                      add_episode=AsyncMock())
            personal = [SimpleNamespace(uuid='personal', group_id='g', expired_at=None)]
            with (
                patch.object(agent, 'ChatOpenAI', FakeModel),
                patch.dict(agent.os.environ, {'DEEPSEEK_API_KEY': 'offline-test'}),
                patch.object(agent.EntityEdge, 'get_by_node_uuid',
                             new=AsyncMock(return_value=personal)),
            ):
                graph = agent.build_agent(graphiti, 'g', 'brand', False, ['catalog'],
                                          memory_recall=enabled, read_only=True)
                output = await graph.ainvoke({
                    'messages': [HumanMessage(content='Recommend shoes?')],
                    'user_name': 'jess', 'user_node_uuid': 'jess-id',
                }, config={'configurable': {'thread_id': 'fresh-thread'}})
            graphiti.add_episode.assert_not_awaited()
            self.assertEqual(len(output['messages']), 4)
            prompts[enabled] = captured[0][0].content
            results[enabled] = [m.content for m in output['messages'] if isinstance(m, ToolMessage)]
            if not enabled:
                captured.clear()
                with patch.object(agent.EntityEdge, 'get_by_node_uuid',
                                  new=AsyncMock()) as personal_lookup:
                    await agent.run_turn(
                        graph, {'configurable': {'thread_id': 'history-thread'}},
                        'jess-id', 'Recommend shoes?',
                        history=[{'role': 'user', 'content': 'My original budget is 100 USD.'}],
                    )
                    personal_lookup.assert_not_awaited()
                self.assertEqual(captured[0][1].content, 'My original budget is 100 USD.')
                self.assertEqual(captured[1][1].content, 'My original budget is 100 USD.')
                graphiti.add_episode.assert_not_awaited()
        self.assertIn('Jess prefers black shoes', prompts[True])
        self.assertNotIn('Jess prefers black shoes', prompts[False])
        self.assertEqual(
            prompts[True].replace(
                '- Jess prefers black shoes, budget 100.', 'No matching facts found.'
            ),
            prompts[False],
        )
        self.assertEqual(results[True], results[False])


if __name__ == '__main__':
    unittest.main()
