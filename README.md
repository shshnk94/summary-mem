# summary-mem

A custom long-term memory mechanism for LLM conversations, designed to plug
into the [`memLLM`](../memLLM) benchmark.

- **Hierarchical / incremental summaries** (mem0-inspired): turns are condensed
  into a tree of rolling summaries that retrieval can enter at any granularity.

It is **self-contained** — it does not depend on the `hipporag` or `mem0`
packages — and reuses memLLM's conventions (OpenRouter client,
`openai/gpt-4o-mini`, `qwen/qwen3-embedding-8b`).

> Status: **skeleton**. Modules contain the intended structure, signatures, and
> `TODO`s; method bodies raise `NotImplementedError`.

## Layout

```
src/summary_mem/
  memory.py       # SummaryMemory(Memory) — the public mechanism (ingest/build_context)
  summarize.py    # hierarchical + incremental summarization
  store.py        # dense vector store over summary nodes
  clients.py      # OpenRouter LLM + embedding helpers
  config.py       # model ids, endpoints, retrieval knobs
  prompts/        # summarization prompt templates
tests/
  test_memory.py  # interface-contract smoke test
```

## Interface

`SummaryMemory` implements memLLM's `Memory` ABC:

| method | purpose |
| --- | --- |
| `ingest(turns)` | append turns; summarize, embed, store |
| `build_context(question, top_k)` | return `(context_str, turn_ids \| None)` |
| `clear()` | reset state for the next conversation |
| `cleanup()` | release on-disk per-conversation cache |

## Testing recall (LongMemEval-style)

`summary_mem.eval` checks whether the mechanism actually *keeps the needle*:
it **indexes** a multi-session haystack into memory, **answers** a question
whose evidence lives in one of those sessions using only the recalled running
summary, then **judges** the answer against the gold answer with an LLM — the
[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) `index` → `rag_qa` flow,
specialized for [LongMemEval](https://github.com/xiaowu0162/LongMemEval) recall.

```python
from summary_mem import MemoryEvaluator
from summary_mem.clients import get_chat_client

ev = MemoryEvaluator(get_chat_client(), conversation_id="q1")
ev.index(haystack_sessions, dates=haystack_dates)   # list of [{"role","content"}, ...]
result = ev.rag_qa(question, question_date=date, gold_answer=gold)
print(result.answer, result.correct)
```

Run it from the CLI — a built-in toy haystack with no data, or a real
LongMemEval file:

```bash
python -m summary_mem.eval                              # quick-start toy run
python -m summary_mem.eval --data longmemeval_s.json --limit 50 --out report.json
```

The CLI prints overall accuracy plus a per-`question_type` breakdown.

> Note: this mechanism keeps a single rolling summary per speaker (no per-turn
> retrieval), so it reports answer correctness — not turn/session retrieval
> recall.

## Registering with memLLM

In `memLLM/src/memory/__init__.py`:

```python
from summary_mem import SummaryMemory
MEMORIES["summary_mem"] = SummaryMemory
```

Then run, e.g.:

```bash
python -m src.generate --memory summary_mem --benchmarks <name> --topk 10
```

## Environment

Set `OPENROUTER_API_KEY` (read from the environment / `.env`, as in memLLM).
