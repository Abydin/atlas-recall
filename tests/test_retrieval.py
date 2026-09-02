from unittest.mock import patch

from atlas_recall.corpus import load_corpus
from atlas_recall.retrieval import top_pointers, format_pointer_block, score_docs


def test_load_corpus_reads_all_notes(notes_dir):
    docs = load_corpus(notes_dir)
    names = {d["name"] for d in docs}
    assert "feedback-no-em-dashes" in names
    assert "MEMORY" not in names  # excluded even if present


def test_query_with_no_dense_and_no_external_services(cfg):
    """The whole product decision: this must work with chromadb absent and
    no Ollama running, and it must not raise."""
    pointers = top_pointers("em dash outward text", cfg)
    assert pointers, "expected at least one hit on a clear keyword match"
    assert pointers[0]["name"] == "feedback-no-em-dashes"


def test_off_topic_query_returns_nothing(cfg):
    pointers = top_pointers("what should I eat for lunch", cfg)
    assert pointers == []


def test_empty_corpus_never_raises(tmp_path, cfg):
    cfg.notes_dir = str(tmp_path / "empty")
    pointers = top_pointers("anything at all", cfg)
    assert pointers == []


def test_wikilink_expansion_pulls_in_high_priority_neighbor(cfg):
    """Contract: a top hit's wikilinked hard-rule/high-priority neighbor
    gets promoted into the result even when it's ranked below top_k on its
    own -- as long as it's independently present in the scored candidate
    set. Pin `score_docs`'s return order directly so this test exercises
    the promotion logic itself rather than depending on BM25 term-weight
    tuning to happen to produce a particular order."""
    docs = load_corpus(cfg.notes_dir)
    by_name = {d["name"]: d for d in docs}
    top_doc = by_name["feedback-verify-before-claiming"]  # links to no-em-dashes
    neighbor = by_name["feedback-no-em-dashes"]            # hard-rule priority
    cfg.top_k = 1  # forces the neighbor OUT of `top`, into expansion-only territory

    with patch(
        "atlas_recall.retrieval.score_docs",
        return_value=[(top_doc, 0.05), (neighbor, 0.04)],
    ):
        pointers = top_pointers("anything", cfg, docs=docs)

    names = {p["name"] for p in pointers}
    assert names == {"feedback-verify-before-claiming", "feedback-no-em-dashes"}


def test_format_pointer_block_empty():
    assert format_pointer_block([]) == ""


def test_score_docs_never_raises_when_dense_enabled_but_unreachable(cfg):
    """dense_enabled=True but nothing is actually running -- must degrade
    to BM25-only, not crash."""
    cfg.dense_enabled = True
    docs = load_corpus(cfg.notes_dir)
    scored = score_docs("em dash", docs, cfg)
    assert scored  # BM25 alone still finds it
