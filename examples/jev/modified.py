r"""Run the Jev example's timeline with DeepSeek, Ollama, and Neo4j.

Run from the repository root:
    .\.venv\Scripts\python.exe examples\jev\modified.py

This compares Graphiti's ordinary DeepSeek-based deduplication and temporal
resolution with the separate Jev-based example. It does not call the Jev API.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Reuse the exact provider setup that already works for the Neo4j quickstart.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quickstart'))
from quickstart_deepseek_ollama_neo4j import create_graphiti  # noqa: E402

from graphiti_core.edges import EntityEdge  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType  # noqa: E402


EPISODES = [
    {
        'content': (
            'Alice Chen is a software engineer at Acme. '
            'Alice manages the customer billing platform.'
        ),
        'reference_time': datetime(2025, 1, 1, tzinfo=timezone.utc),
        'description': 'Alice Chen joins Acme',
    },
    {
        'content': (
            'Ms. Chen works for Acme as a software engineer. '
            'Alice is responsible for managing the customer billing platform.'
        ),
        'reference_time': datetime(2025, 1, 4, tzinfo=timezone.utc),
        'description': 'Ms. Chen continues at Acme',
    },
    {
        'content': (
            'Alice Chen manages the customer billing platform. '
            'She is the only person who manages it.'
        ),
        'reference_time': datetime(2025, 1, 7, tzinfo=timezone.utc),
        'description': 'Alice Chen manages the customer billing platform',
    },
    {
        'content': (
            'Priya Patel took over the customer billing platform from Alice Chen. '
            'Priya now manages it instead of Alice.'
        ),
        'reference_time': datetime(2025, 1, 10, tzinfo=timezone.utc),
        'description': 'Priya Patel takes over the billing platform',
    },
]


async def main() -> None:
    group_id = f'deepseek-jev-demo-{uuid4().hex}'
    graphiti = await create_graphiti()
    print(f'Experiment group_id: {group_id}')

    try:
        for index, episode in enumerate(EPISODES, start=1):
            result = await graphiti.add_episode(
                name=f'DeepSeek Jev comparison episode {index}',
                episode_body=episode['content'],
                source=EpisodeType.text,
                source_description=episode['description'],
                reference_time=episode['reference_time'],
                group_id=group_id,
            )
            print(f'\n--- Episode {index}: {episode["description"]} ---')
            print(f'  Entities ({len(result.nodes)}):')
            for node in result.nodes:
                print(f'    - {node.name} (UUID: {node.uuid})')
            print(f'  Facts ({len(result.edges)}):')
            for edge in result.edges:
                print(f'    - {edge.fact}')

        edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id])
        nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])

        print('\nAll entity facts and temporal fields:')
        for edge in edges:
            valid_at = edge.valid_at.isoformat() if edge.valid_at else 'unknown'
            invalid_at = edge.invalid_at.isoformat() if edge.invalid_at else 'none'
            expired_at = edge.expired_at.isoformat() if edge.expired_at else 'none'
            print(f'  - {edge.fact}')
            print(f'    valid_at={valid_at}; invalid_at={invalid_at}; expired_at={expired_at}')

        print('\nEntity nodes:')
        for node in nodes:
            print(f'  - {node.name} (UUID: {node.uuid})')
        print(f'\nExperiment group_id: {group_id}')
        print('Compare Alice and Priya facts above to see whether the transfer was resolved.')
    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
