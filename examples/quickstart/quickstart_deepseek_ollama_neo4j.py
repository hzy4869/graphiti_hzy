"""Minimal Graphiti run: DeepSeek extraction, local Ollama embeddings, Neo4j storage.

Run from the repository root after filling .env and pulling bge-m3 in Ollama.
"""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import dotenv_values
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI
from pydantic import ValidationError

for key, value in dotenv_values().items():
    if value:
        os.environ.setdefault(key, value)
os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
os.environ.setdefault('SEMAPHORE_LIMIT', '2')
for proxy_var in ('NO_PROXY', 'no_proxy'):
    bypass = os.environ.get(proxy_var, '')
    os.environ[proxy_var] = ','.join(filter(None, (bypass, 'localhost', '127.0.0.1')))

for required_key in ('NEO4J_URI', 'NEO4J_PASSWORD', 'DEEPSEEK_API_KEY'):
    if not os.environ.get(required_key):
        raise RuntimeError(f'Set {required_key} in the repository-root .env file before running.')

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.driver.neo4j_driver import Neo4jDriver  # noqa: E402
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402


class DeepSeekValidatedClient(OpenAIGenericClient):
    """Validate JSON-mode replies before Graphiti consumes them.

    DeepSeek can occasionally put field values under a top-level ``properties``
    key after seeing a JSON schema. Accept that unambiguous wrapper; otherwise
    ask once more for the required shape instead of failing mid-ingestion.
    """

    async def generate_response(self, messages, response_model=None, **kwargs):
        for attempt in range(2):
            attempt_messages = [message.model_copy(deep=True) for message in messages]
            if attempt:
                attempt_messages[-1].content += (
                    '\nReturn the requested JSON field values directly at the top level. '
                    'Do not return a JSON Schema or wrap the values in "properties".'
                )
            result = await super().generate_response(
                attempt_messages, response_model=response_model, **kwargs
            )
            if response_model is None:
                return result
            try:
                response_model.model_validate(result)
                return result
            except ValidationError:
                nested = result.get('properties') if isinstance(result, dict) else None
                if isinstance(nested, dict):
                    try:
                        response_model.model_validate(nested)
                        return nested
                    except ValidationError:
                        pass
            if attempt == 0:
                print(f'Retrying malformed DeepSeek JSON for {response_model.__name__}.')
        raise ValueError(f'DeepSeek returned invalid JSON fields for {response_model.__name__}.')


class CrossEncoderNotConfigured(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        raise RuntimeError('This example uses RRF search; cross-encoder ranking is not configured.')


class SequentialOllamaEmbedder(OpenAIEmbedder):
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors = []
        for index, text in enumerate(input_data_list, start=1):
            try:
                vectors.append(await self.create(text))
            except Exception:
                print(
                    f'Local embedding failed on item {index}/{len(input_data_list)} '
                    f'({len(text)} characters).'
                )
                raise
        return vectors


async def create_graphiti() -> Graphiti:
    uri = os.environ['NEO4J_URI']
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ['NEO4J_PASSWORD']
    database = os.environ.get('NEO4J_DATABASE')
    if not database:
        async with AsyncGraphDatabase.driver(uri, auth=(user, password)) as discovery_driver:
            _, summary, _ = await discovery_driver.execute_query('RETURN 1', database_=None)
            database = summary.database
        if not database:
            raise RuntimeError('Could not detect the Neo4j database name; set NEO4J_DATABASE in .env.')
    print(f'Using Neo4j database: {database}')

    graph_driver = Neo4jDriver(uri, user, password, database=database)
    graphiti = Graphiti(
        graph_driver=graph_driver,
        llm_client=DeepSeekValidatedClient(
            config=LLMConfig(
                api_key=os.environ['DEEPSEEK_API_KEY'],
                model='deepseek-flash',
                small_model='deepseek-flash',
                base_url='https://api.deepseek.com',
            ),
            structured_output_mode='json_object',
        ),
        embedder=SequentialOllamaEmbedder(
            config=OpenAIEmbedderConfig(
                api_key='ollama',
                embedding_model='bge-m3',
                embedding_dim=1024,
                base_url='http://127.0.0.1:11434/v1',
            ),
            client=AsyncOpenAI(
                api_key='ollama',
                base_url='http://127.0.0.1:11434/v1',
                timeout=120.0,
                max_retries=5,
            ),
        ),
        cross_encoder=CrossEncoderNotConfigured(),
        max_coroutines=2,
    )

    try:
        # Neo4jDriver starts index creation during construction. Wait for that
        # task instead of starting a second, concurrent index-creation pass.
        if graph_driver._init_task is not None:
            await graph_driver._init_task
        else:
            await graphiti.build_indices_and_constraints()
        try:
            await graphiti.embedder.create('Graphiti startup probe')
        except Exception as exc:
            response = getattr(exc, 'response', None)
            if response is not None:
                print(
                    f'Local embedding HTTP {response.status_code}; '
                    f'server={response.headers.get("server", "unknown")}; '
                    f'via={response.headers.get("via", "none")}.'
                )
            raise
        print('Local bge-m3 embedding is ready.')
    except Exception:
        await graphiti.close()
        raise
    return graphiti


async def main() -> None:
    graphiti = await create_graphiti()
    try:
        await graphiti.add_episode(
            name='local-embedding-smoke-test',
            episode_body='Alice joined Acme in 2023. Bob joined Acme in 2024.',
            source=EpisodeType.text,
            source_description='Graphiti smoke test',
            reference_time=datetime.now(timezone.utc),
        )
        results = await graphiti.search('Who joined Acme in 2023?')
        for result in results:
            print(result.fact)
        if not results:
            print('Episode was added, but this query returned no facts.')
    finally:
        await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())
