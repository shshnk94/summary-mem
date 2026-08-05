"""summary-mem: a hierarchical-summary memory mechanism.

Drop-in for the memLLM benchmark. To register it there, add to
`memLLM/src/memory/__init__.py`:

    from summary_mem import SummaryMemory
    MEMORIES["summary_mem"] = SummaryMemory
"""

from .memory import InContextMemory, SummaryMemory

__all__ = ["InContextMemory", "SummaryMemory"]

# The evaluation harness (LongMemEval-style recall test) lives in
# ``summary_mem.eval`` and is imported from there to keep ``python -m
# summary_mem.eval`` free of double-import warnings:
#     from summary_mem.eval import MemoryEvaluator
