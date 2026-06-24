"""Hierarchical / incremental summarization.

The summary layer is the namesake of this package. Leaf summaries condense
windows of turns; higher levels recursively condense lower ones, giving a
tree of progressively coarser memory that retrieval can enter at any level.

Inspired by mem0's incremental consolidation (add / update / merge of memory
as new turns arrive) rather than a single static pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SummaryNode:
    """One node in the summary hierarchy."""
    node_id: str
    text: str
    level: int                                   # 0 = leaf (turn window)
    child_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)


def summarize_window(turns: list[dict]) -> SummaryNode:
    """Condense a window of turns into a single leaf summary node."""
    # TODO: prompt the LLM with prompts.summarization.
    raise NotImplementedError


def consolidate(existing: list[SummaryNode], new: list[SummaryNode]) -> list[SummaryNode]:
    """mem0-style incremental update: merge/update/append summaries as turns arrive."""
    # TODO: decide what to keep, merge, or supersede.
    raise NotImplementedError


def build_hierarchy(leaves: list[SummaryNode]) -> list[SummaryNode]:
    """Recursively roll leaf summaries up into higher-level summary nodes."""
    # TODO: group + summarize level-by-level until a single root remains.
    raise NotImplementedError
