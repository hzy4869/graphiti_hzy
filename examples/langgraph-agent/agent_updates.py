r"""Sequential preference-update experiment copied from agent_modified.py.

Run from the repository root:
    .\.venv\Scripts\python.exe examples\langgraph-agent\agent_updates.py

Prepare updates with --prepare-history history_updates.json. Each dated user
turn becomes a separate Graphiti episode. Graphiti and full-history evaluation
receive the same dated source messages. Ground truth is never sent to the LLM.

The default run asks DeepSeek to extract complete product inventory from JSON,
recalls only customer-linked facts, and requires a shoe lookup for product
questions. This differs from agent.ipynb's unconstrained extraction and
optional tool call, so keep earlier baseline logs for comparisons.

Use --interactive to keep chatting, --demo-memory for two additional turns,
or --resume-group GROUP_ID to recall Graphiti facts from a previous process.
Use --catalog-fallback to provide exact sizes from the original JSON catalog
when graph extraction misses them; the default preserves the notebook's tool.
Use --seed-size-facts to write catalog size facts through Graphiti's add_triplet
API before the agent runs, so the original graph-search tool can retrieve them.
MemorySaver is process-local.

Use --memory-recall on|off (default: on) to control customer-fact retrieval.
Both modes search only catalog-sourced edges in the shoe tool. For paired
evaluation, use --resume-group GROUP_ID --read-only --question QUESTION in
separate processes; this starts empty message history and does not save answers.
Without --read-only, conversation writing remains enabled in both modes.

Prepare a fixed JSON transcript with --prepare-history FILE, then evaluate the
same group using --memory-mode graphiti|history --history-file FILE --question TEXT.
Evaluation is read-only and single-turn. --results-file FILE appends metrics JSONL.
Token counts cover all agent LLM calls; turn latency excludes setup and ingestion.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import operator
import os
import re
import sys
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from experiment_usage import PreparationUsage, group_preparation, normalize_usage, read_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quickstart'))
from quickstart_deepseek_ollama_neo4j import create_graphiti  # noqa: E402

from graphiti_core.edges import EntityEdge  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # noqa: E402
from graphiti_core.search.search_config_recipes import (  # noqa: E402
    NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
)
from graphiti_core.search.search_filters import SearchFilters  # noqa: E402


PRODUCT_FILE = Path(__file__).resolve().parents[1] / 'data' / 'manybirds_products.json'
USER_NAME = 'jess'
PRODUCT_EXTRACTION_INSTRUCTIONS = (
    'This episode is a product catalog record. Keep the product named in title and '
    'the vendor Manybirds as distinct entities. Extract an inventory relationship '
    'between Manybirds and this exact product. Its fact must enumerate every '
    'variants[].option1 size together with variants[].available status. A false '
    'status means out of stock. Preserve all sizes even when every variant is '
    'out of stock. Do not copy sizes from a different product.'
)
PRODUCT_REQUEST_TERMS = re.compile(
    r'\b(?:shoe|shoes|sneaker|sneakers|runner|runners|product|size|sizes|price|'
    r'cost|stock|available|availability|recommend|suggest)\b',
    re.IGNORECASE,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str
    user_node_uuid: str
    call_metrics: Annotated[list, operator.add]


def load_history(path: str) -> tuple[list[dict], str]:
    """Read fixed user/assistant turns; disallow hidden system or tool instructions."""
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    messages = data.get('messages') if isinstance(data, dict) else None
    if not isinstance(messages, list) or not messages:
        raise ValueError('History must contain a nonempty messages list.')
    cleaned = []
    previous_time = None
    for index, message in enumerate(messages):
        expected = 'user' if index % 2 == 0 else 'assistant'
        if (
            not isinstance(message, dict) or message.get('role') != expected
            or not isinstance(message.get('content'), str) or not message['content'].strip()
        ):
            raise ValueError(f'History turn {index} must be a nonempty {expected} message.')
        timestamp = parse_update_time(message.get('timestamp'))
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError('Update timestamps must be strictly increasing.')
        previous_time = timestamp
        cleaned.append({
            'role': expected,
            'content': f'[{timestamp.isoformat()}] {message["content"]}',
        })
    canonical = json.dumps(cleaned, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return cleaned, hashlib.sha256(canonical).hexdigest()


def parse_update_time(value) -> datetime:
    if not isinstance(value, str):
        raise ValueError('Every update message requires an ISO timestamp with timezone.')
    try:
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'Invalid timestamp: {value}') from exc
    if timestamp.tzinfo is None:
        raise ValueError('Update timestamps must include a timezone.')
    timestamp = timestamp.astimezone(timezone.utc)
    if timestamp > datetime.now(timezone.utc):
        raise ValueError('Update timestamps must not be in the future.')
    return timestamp


async def prepare_updates(graphiti, group_id: str, path: str, tracker=None) -> None:
    messages, digest = load_history(path)
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))['messages']
    episodes = await EpisodicNode.get_by_group_ids(graphiti.driver, [group_id])
    existing_names = {episode.name for episode in episodes}
    total = (len(messages) + 1) // 2
    for index in range(0, len(messages), 2):
        number = index // 2 + 1
        if tracker:
            tracker.item = f'update-{number}'
        name = (
            f'Experiment history {digest}' if number == total
            else f'Update history {digest} turn {number}'
        )
        if name in existing_names:
            print(f'Already prepared update {number}/{total}')
            continue
        batch = messages[index:index + 2]
        result = await graphiti.add_episode(
            name=name,
            episode_body='\n'.join(
                f'{USER_NAME if m["role"] == "user" else "SalesBot"}: {m["content"]}'
                for m in batch
            ),
            source=EpisodeType.message,
            source_description='Experiment conversation updates',
            reference_time=parse_update_time(raw[index]['timestamp']),
            group_id=group_id,
            custom_extraction_instructions=(
                'Extract customer-stated shoe size, budget, color preference and material '
                'restrictions as separate facts connected to the customer. A change replaces '
                'the previous value only for the explicitly changed dimension; other '
                'requirements continue to apply. Preserve exact values and the stated date. '
                'A replacement material restriction list replaces the whole former list. '
                'Assistant acknowledgements add no new customer preferences. Never infer '
                'that all old requirements were cancelled when one dimension changes.'
            ),
        )
        print(f'Prepared update {number}/{total}: '
              f'{len(result.edges)} returned fact(s)', flush=True)
    print(f'History prepared: {digest}; {len(messages)} message(s)', flush=True)
    nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
    user = next((node for node in nodes if node.name.casefold() == USER_NAME), None)
    if user:
        edges = await EntityEdge.get_by_node_uuid(graphiti.driver, user.uuid)
        print('Customer graph facts after updates (inspect stale/current dates):', flush=True)
        for edge in edges:
            print(f'  {edge.fact}; valid_at={edge.valid_at}; '
                  f'invalid_at={edge.invalid_at}; expired_at={edge.expired_at}', flush=True)


def write_result(path: str | None, result: dict) -> None:
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')


def facts_to_string(edges) -> str:
    facts = list(dict.fromkeys(edge.fact for edge in edges))
    return '\n'.join(f'- {fact}' for fact in facts) or 'No matching facts found.'


def catalog_size_facts(query: str) -> str:
    """Return exact source-catalog sizes when a query identifies one product."""
    products = json.loads(PRODUCT_FILE.read_text(encoding='utf-8'))['products']
    ignored = {'the', 'and', 'for', 'what', 'which', 'size', 'sizes', 'available', 'shoe'}
    query_words = set(re.findall(r'[a-z0-9]+', query.casefold())) - ignored
    matches = []
    for product in products:
        title_words = set(re.findall(r'[a-z0-9]+', product['title'].casefold()))
        score = len(query_words & title_words)
        matches.append((score, product))
    matches.sort(key=lambda item: item[0], reverse=True)
    if not matches or matches[0][0] < 2:
        return ''
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        return ''
    product = matches[0][1]
    return product_size_statement(product)


def product_size_statement(product: dict) -> str:
    """Turn structured variants into one explicit fact for Graphiti ingestion."""
    variants = product.get('variants', [])
    if not variants:
        return ''
    sizes = ', '.join(
        f'{variant.get("option1", "unknown")} '
        f'({"available" if variant.get("available") else "out of stock"})'
        for variant in variants
    )
    return f'Catalog record for {product["title"]}: sizes {sizes}.'


async def ingest_products(graphiti, group_id: str) -> None:
    products = json.loads(PRODUCT_FILE.read_text(encoding='utf-8'))['products']
    for index, product in enumerate(products):
        body = {key: value for key, value in product.items() if key != 'images'}
        result = await graphiti.add_episode(
            name=product.get('title', f'Product {index}'),
            episode_body=json.dumps(body, ensure_ascii=False),
            source=EpisodeType.json,
            source_description='ManyBirds products',
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
            custom_extraction_instructions=PRODUCT_EXTRACTION_INSTRUCTIONS,
        )
        print(f'Loaded product: {product["title"]}', flush=True)
        variants = product.get('variants', [])
        size_facts = [
            edge.fact for edge in result.edges
            if variants and all(str(variant.get('option1')) in edge.fact for variant in variants)
        ]
        print(f'  All size labels extracted: {bool(size_facts)}', flush=True)


async def seed_size_facts(graphiti, group_id: str) -> None:
    """Write exact catalog sizes as graph edges when extraction omits them."""
    products = json.loads(PRODUCT_FILE.read_text(encoding='utf-8'))['products']
    nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
    episodes = await EpisodicNode.get_by_group_ids(graphiti.driver, [group_id])
    existing = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id])
    nodes_by_name = {node.name.casefold(): node for node in nodes}
    episodes_by_name = {episode.name: episode for episode in episodes}
    vendor = nodes_by_name.get('manybirds')
    if vendor is None:
        raise RuntimeError(f'Could not find Manybirds in group {group_id}.')

    for product in products:
        product_node = nodes_by_name.get(product['title'].casefold())
        if product_node is None:
            raise RuntimeError(f'Could not find product {product["title"]!r} in group {group_id}.')
        fact = product_size_statement(product)
        if not fact or any(edge.fact == fact for edge in existing):
            continue
        episode = episodes_by_name.get(product['title'])
        edge = EntityEdge(
            source_node_uuid=vendor.uuid,
            target_node_uuid=product_node.uuid,
            name='OFFERS_SIZES',
            fact=fact,
            group_id=group_id,
            episodes=[episode.uuid] if episode else [],
            created_at=datetime.now(timezone.utc),
            valid_at=datetime.now(timezone.utc),
        )
        await graphiti.add_triplet(vendor, edge, product_node, episode=episode)
        print(f'Seeded size fact: {product["title"]}', flush=True)


async def find_node_uuid(graphiti, group_id: str, name: str) -> str:
    result = await graphiti.search_(
        name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS, group_ids=[group_id]
    )
    for node in result.nodes:
        if node.name.casefold() == name.casefold():
            return node.uuid
    for node in await EntityNode.get_by_group_ids(graphiti.driver, [group_id]):
        if node.name.casefold() == name.casefold():
            return node.uuid
    raise RuntimeError(f'Could not find the {name!r} entity in group {group_id}.')


async def catalog_edge_uuids(graphiti, group_id: str) -> list[str]:
    """Exclude conversation-derived facts from product retrieval in both modes."""
    episodes = await EpisodicNode.get_by_group_ids(graphiti.driver, [group_id])
    catalog_episodes = {
        episode.uuid for episode in episodes
        if episode.source_description == 'ManyBirds products'
    }
    edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id])
    return [
        edge.uuid for edge in edges
        if edge.episodes and set(edge.episodes).issubset(catalog_episodes)
    ]


async def recall_customer_facts(graphiti, state: State, group_id: str, enabled: bool):
    if not enabled:
        return []
    last_message = state['messages'][-1]
    speaker = 'SalesBot' if isinstance(last_message, AIMessage) else state['user_name']
    personal_edges = await EntityEdge.get_by_node_uuid(graphiti.driver, state['user_node_uuid'])
    personal_uuids = [
        edge.uuid for edge in personal_edges
        if edge.group_id == group_id and edge.expired_at is None
        and (edge.invalid_at is None or edge.invalid_at > datetime.now(timezone.utc))
        and (edge.valid_at is None or edge.valid_at <= datetime.now(timezone.utc))
    ]
    if not personal_uuids:
        return []
    return await graphiti.search(
        f'{speaker}: {last_message.content}',
        center_node_uuid=state['user_node_uuid'],
        group_ids=[group_id],
        num_results=5,
        search_filter=SearchFilters(edge_uuids=personal_uuids),
    )


def build_agent(
    graphiti,
    group_id: str,
    manybirds_node_uuid: str,
    catalog_fallback: bool,
    catalog_uuids: list[str],
    memory_recall: bool = True,
    read_only: bool = False,
):
    @tool
    async def get_shoe_data(query: str) -> str:
        """Search the graphiti graph for information about shoes"""
        edges = await graphiti.search(
            query,
            center_node_uuid=manybirds_node_uuid,
            group_ids=[group_id],
            num_results=10,
            search_filter=SearchFilters(edge_uuids=catalog_uuids),
        )
        result = facts_to_string(edges)
        if catalog_fallback:
            source_sizes = catalog_size_facts(query)
            if source_sizes:
                result = f'{result}\n[Original product catalog] {source_sizes}'
        print(f'  [Agent tool] get_shoe_data({query!r})\n{result}', flush=True)
        return result

    tools = [get_shoe_data]
    tool_node = ToolNode(tools)
    model = ChatOpenAI(
        model='deepseek-flash',
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url='https://api.deepseek.com',
        timeout=120.0,
        max_retries=3,
        extra_body={'thinking': {'type': 'disabled'}},
    )
    llm = model.bind_tools(tools)
    tool_required_llm = model.bind_tools(tools, tool_choice='required')

    async def chatbot(state: State):
        last_message = state['messages'][-1]
        edges = await recall_customer_facts(graphiti, state, group_id, memory_recall)
        remembered = facts_to_string(edges)
        recall_status = 'on' if memory_recall else 'off'
        print(f'  [Graphiti recall: {recall_status}]\n{remembered}', flush=True)
        system = SystemMessage(
            content=(
                'You are a skillfull shoe salesperson working for ManyBirds. '
                'Review information about the user and their prior conversation '
                'below and respond accordingly.\n'
                '        Keep responses short and concise. And remember, always '
                'be selling (and helpful!)\n\n'
                "        Things you'll need to know about the user in order to close a sale:\n"
                "        - the user's shoe size\n"
                '        - any other shoe needs? maybe for wide feet?\n'
                '        - the user\'s preferred colors and styles\n'
                '        - their budget\n\n'
                "        Ensure that you ask the user for the above if you don't already know.\n\n"
                '        Facts about the user and their conversation:\n'
                f'        {remembered}\n\n'
                'Customer facts above are not a complete product catalog. When asked '
                'about a product, use get_shoe_data before answering. Missing recall '
                'does not mean the store lacks the product. Do not infer that a '
                'product is free of an allergen from a partial material description.'
            )
        )
        needs_product_tool = (
            isinstance(last_message, HumanMessage)
            and bool(PRODUCT_REQUEST_TERMS.search(last_message.content))
            and (
                '?' in last_message.content
                or bool(
                    re.search(
                        r'\b(?:recommend|suggest|find|show|check)\b',
                        last_message.content,
                        re.I,
                    )
                )
            )
        )
        chosen_llm = tool_required_llm if needs_product_tool else llm
        model_started = perf_counter()
        response = await chosen_llm.ainvoke([system, *state['messages']])
        usage = response.usage_metadata or {}
        provider_usage = response.response_metadata.get('token_usage') or usage
        call_metric = {
            **normalize_usage(provider_usage),
            'seconds': perf_counter() - model_started,
        }
        if not response.tool_calls:
            user_message = next(
                (message for message in reversed(state['messages'])
                 if isinstance(message, HumanMessage)),
                None,
            )
            if user_message is not None and not read_only:
                await graphiti.add_episode(
                    name='Chatbot Response',
                    episode_body=(
                        f'{state["user_name"]}: {user_message.content}\n'
                        f'SalesBot: {response.content}'
                    ),
                    source=EpisodeType.message,
                    source_description='Chatbot',
                    reference_time=datetime.now(timezone.utc),
                    group_id=group_id,
                    custom_extraction_instructions=(
                        'Preserve customer-stated preferences and allergies as separate '
                        'facts involving the customer. Do not infer that a product is '
                        'free of a material merely because another material is named.'
                    ),
                )
        return {'messages': [response], 'call_metrics': [call_metric]}

    def should_continue(state: State) -> str:
        return 'continue' if state['messages'][-1].tool_calls else 'end'

    builder = StateGraph(State)
    builder.add_node('agent', chatbot)
    builder.add_node('tools', tool_node)
    builder.add_edge(START, 'agent')
    builder.add_conditional_edges('agent', should_continue, {'continue': 'tools', 'end': END})
    builder.add_edge('tools', 'agent')
    return builder.compile(checkpointer=MemorySaver())


async def run_turn(
    graph, config: dict, user_node_uuid: str, question: str,
    history: list[dict] | None = None, metadata: dict | None = None,
    results_file: str | None = None,
) -> dict:
    print(f'\n{USER_NAME}: {question}', flush=True)
    state = {
        'messages': [*(history or []), {'role': 'user', 'content': question}],
        'user_name': USER_NAME,
        'user_node_uuid': user_node_uuid,
    }
    tool_calls = 0
    calls, answers, trace = [], [], []
    started = perf_counter()
    error = None
    try:
        async for event in graph.astream(state, config=config):
            for node_name, value in event.items():
                trace.extend(message.model_dump() for message in value.get('messages', []))
                if node_name == 'agent':
                    calls.extend(value.get('call_metrics', []))
                    message = value['messages'][-1]
                    tool_calls += len(message.tool_calls)
                    if message.content and not message.tool_calls:
                        answers.append(message.content)
                        print(f'SalesBot: {message.content}', flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        def tokens(key):
            values = [call.get(key) for call in calls]
            return sum(values) if values and all(v is not None for v in values) else None

        result = {
            **(metadata or {}),
            'run_id': uuid4().hex,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'question': question,
            'answer': '\n'.join(answers),
            'status': 'error' if error else 'ok',
            'error': error,
            'turn_seconds': perf_counter() - started,
            'llm_seconds': sum(call['seconds'] for call in calls),
            'llm_calls': len(calls),
            'tool_calls': tool_calls,
            'input_tokens': tokens('input_tokens'),
            'output_tokens': tokens('output_tokens'),
            'cache_hit_tokens': tokens('cache_hit_tokens'),
            'cache_miss_tokens': tokens('cache_miss_tokens'),
            'call_metrics': calls,
            'trace': trace,
        }
        write_result(results_file, result)
        print(f'Turn seconds: {result["turn_seconds"]:.3f}; '
              f'input tokens: {result["input_tokens"]}; '
              f'output tokens: {result["output_tokens"]}', flush=True)
    print(f'Tool calls this turn: {tool_calls}', flush=True)
    return result


async def main(
    interactive: bool,
    resume_group: str | None,
    demo_memory: bool,
    catalog_fallback: bool,
    seed_sizes: bool,
    question: str | None,
    memory_recall: bool = True,
    read_only: bool = False,
    memory_mode: str | None = None,
    history_file: str | None = None,
    prepare_history: str | None = None,
    results_file: str | None = None,
    prep_usage_file: str = 'results/update_preparation_usage.jsonl',
) -> None:
    logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)
    group_id = resume_group or f'shoebot-deepseek-{uuid4().hex}'
    tracker = None
    if memory_mode:
        group_preparation(prep_usage_file, group_id)
    if prepare_history and resume_group:
        previous = [row for row in read_records(prep_usage_file)
                    if row.get('group_id') == group_id]
        finished = {row['phase'] for row in previous if row['event'] == 'phase_complete'}
        if not {'catalog', 'user_creation'}.issubset(finished):
            raise ValueError('Cannot resume unmetered preparation. Start a new group.')
    graphiti = await create_graphiti()
    print(f'Experiment group_id: {group_id}', flush=True)
    try:
        if prepare_history or not resume_group:
            tracker = PreparationUsage(prep_usage_file, group_id)
            tracker.attach(graphiti)
        if not resume_group:
            tracker.begin_phase('catalog')
            await ingest_products(graphiti, group_id)
            tracker.finish_phase()
            tracker.begin_phase('user_creation')
            await graphiti.add_episode(
                name='User Creation',
                episode_body=f'{USER_NAME} is interested in buying a pair of shoes',
                source=EpisodeType.text,
                source_description='SalesBot',
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
            )
            tracker.finish_phase()
        else:
            print('Resumed Graphiti group; LangGraph conversation state starts empty.')

        if prepare_history:
            tracker.begin_phase('customer_updates')
            await prepare_updates(graphiti, group_id, prepare_history, tracker)
            tracker.finish_phase()
            tracker.record('preparation_complete')
            print(f'Preparation API usage ledger: {tracker.path}')
            return

        history, digest = (load_history(history_file) if history_file else ([], None))
        if digest:
            episodes = await EpisodicNode.get_by_group_ids(graphiti.driver, [group_id])
            if not any(e.name == f'Experiment history {digest}' for e in episodes):
                raise ValueError('This history was not prepared in the selected group. '
                                 'Run --prepare-history with the same file first.')
        mode = memory_mode or ('graphiti' if memory_recall else 'none')
        memory_recall = mode == 'graphiti'

        if seed_sizes:
            await seed_size_facts(graphiti, group_id)

        user_node_uuid = await find_node_uuid(graphiti, group_id, USER_NAME)
        manybirds_node_uuid = await find_node_uuid(graphiti, group_id, 'Manybirds')
        catalog_uuids = await catalog_edge_uuids(graphiti, group_id)
        if not catalog_uuids:
            raise RuntimeError('No catalog-only graph facts found; ingest products first.')
        graph = build_agent(
            graphiti, group_id, manybirds_node_uuid, catalog_fallback,
            catalog_uuids, memory_recall, read_only,
        )
        print(f'Customer memory recall: {"on" if memory_recall else "off"}')
        print(f'Conversation writing: {"off" if read_only else "on"}')
        print(f'Catalog-only search scope: {len(catalog_uuids)} fact(s)')
        print(f'Memory mode: {mode}; history messages injected: '
              f'{len(history) if mode == "history" else 0}')
        config = {'configurable': {'thread_id': uuid4().hex}}
        if resume_group:
            questions = [
                'What do you remember about my shoe preferences and allergies? '
                'Can you suggest a black shoe with available sizes?'
            ]
        else:
            questions = ['What sizes do the TinyBirds Wool Runners in Natural Black come in?']
        if question:
            questions = [question]
        if demo_memory:
            questions.extend([
                'I am allergic to wool and I prefer black shoes.',
                'What shoe would you recommend for me, and what sizes are available?',
            ])
        metadata = {
            'group_id': group_id, 'memory_mode': mode, 'read_only': read_only,
            'experiment': 'sequential_preference_updates',
            'preparation_usage_file': str(Path(prep_usage_file).resolve()),
            'history_sha256': digest, 'model': 'deepseek-flash',
            'base_url': 'https://api.deepseek.com',
            'catalog_fact_count': len(catalog_uuids),
            'catalog_scope_sha256': hashlib.sha256(
                '\n'.join(sorted(catalog_uuids)).encode('utf-8')
            ).hexdigest(),
            'history_message_count': len(history) if mode == 'history' else 0,
        }
        for index, question in enumerate(questions):
            await run_turn(
                graph, config, user_node_uuid, question,
                history if mode == 'history' and index == 0 else None,
                metadata, results_file,
            )
        print(f'\nDemo completed: {len(questions)} turn(s)', flush=True)

        if interactive:
            print('\nContinue chatting; enter /exit to finish.', flush=True)
            while True:
                question = (await asyncio.to_thread(input, f'{USER_NAME}: ')).strip()
                if question.lower() in {'/exit', 'exit', 'quit'}:
                    break
                if question:
                    await run_turn(graph, config, user_node_uuid, question)
    finally:
        if tracker:
            tracker.detach()
        await graphiti.close()
        print(f'\nExperiment group_id: {group_id}')
        print('Connections closed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interactive', action='store_true', help='Keep chatting after the demo.')
    parser.add_argument(
        '--demo-memory', action='store_true', help='Run two extra turns to show user memory.'
    )
    parser.add_argument(
        '--catalog-fallback',
        action='store_true',
        help='Add exact size and stock data from the source catalog to shoe-tool results.',
    )
    parser.add_argument(
        '--seed-size-facts',
        action='store_true',
        help='Write exact catalog size facts into Graphiti before querying.',
    )
    parser.add_argument('--question', help='Replace the default first question.')
    parser.add_argument('--memory-mode', choices=('graphiti', 'history', 'none'),
                        help='Single-turn read-only evaluation mode on a prepared group.')
    parser.add_argument('--history-file', help='Fixed prior messages JSON for paired evaluation.')
    parser.add_argument('--prepare-history', help='Ingest prior messages JSON, then exit.')
    parser.add_argument('--results-file', help='Append per-turn metrics and trace as JSONL.')
    parser.add_argument(
        '--prep-usage-file', default='results/update_preparation_usage.jsonl',
        help='Durable ledger for ALL Graphiti ingestion provider calls.',
    )
    parser.add_argument(
        '--memory-recall', choices=('on', 'off'), default=None,
        help='Enable or disable customer-fact retrieval; default: on.',
    )
    parser.add_argument(
        '--read-only', action='store_true',
        help='Evaluate a resumed group without saving replies or reloading products.',
    )
    parser.add_argument(
        '--resume-group',
        metavar='GROUP_ID',
        help='Recall a previous Graphiti group without loading products again.',
    )
    args = parser.parse_args()
    for history_path in (args.history_file, args.prepare_history):
        if history_path:
            try:
                load_history(history_path)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
    if args.memory_mode:
        if (
            args.memory_recall or not args.resume_group or not args.question
            or args.interactive or args.demo_memory or args.seed_size_facts
            or args.catalog_fallback or args.prepare_history
        ):
            parser.error('--memory-mode requires --resume-group and --question; '
                         'do not combine it with demo, ingestion or recall flags.')
        if args.memory_mode in ('history', 'graphiti') and not args.history_file:
            parser.error('Paired graphiti/history evaluation requires --history-file.')
        args.read_only = True
    elif args.history_file:
        parser.error('--history-file requires --memory-mode.')
    if args.prepare_history and (
        args.read_only or args.question or args.interactive or args.demo_memory
        or args.seed_size_facts or args.catalog_fallback
    ):
        parser.error('--prepare-history only ingests history; do not combine with chat options.')
    if args.read_only and (
        not args.resume_group or args.seed_size_facts or args.interactive or args.demo_memory
    ):
        parser.error('--read-only requires --resume-group and a single evaluation turn.')
    asyncio.run(
        main(
            args.interactive,
            args.resume_group,
            args.demo_memory,
            args.catalog_fallback,
            args.seed_size_facts,
            args.question,
            args.memory_recall != 'off',
            args.read_only,
            args.memory_mode,
            args.history_file,
            args.prepare_history,
            args.results_file,
            args.prep_usage_file,
        )
    )
