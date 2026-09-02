"""atlas-recall: deterministic markdown memory for Claude Code.

Public surface:
    from atlas_recall import top_pointers, format_pointer_block, load_config
"""
from .config import Config, load_config
from .retrieval import top_pointers, format_pointer_block, score_docs

__all__ = [
    "Config",
    "load_config",
    "top_pointers",
    "format_pointer_block",
    "score_docs",
]

__version__ = "0.1.0"
