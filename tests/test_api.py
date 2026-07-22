import pytest

import main


def make_hit(cid, text="some code", score=None, **overrides):
    hit = {
        "id": cid,
        "project": "proj",
        "repo": "repo",
        "path": f"src/{cid}.cs",
        "start_line": 1,
        "end_line": 10,
        "file_size_bytes": 100,
        "chunk_tokens_estimate": 10,
        "commit_hash": "abc",
        "commit_date": "2026-01-01T00:00:00Z",
        "text": text,
    }
    if score is not None:
        hit["score"] = score
    hit.update(overrides)
    return hit


class TestTokenizeQuery:
    def test_identifiers_extracted_and_or_joined(self):
        assert main.tokenize_query("TestAuthenticationHandler claims") == "TestAuthenticationHandler OR claims"

    def test_case_insensitive_dedup_keeps_first_spelling(self):
        assert main.tokenize_query("Foo foo FOO bar") == "Foo OR bar"

    def test_single_char_tokens_dropped(self):
        assert main.tokenize_query("a b cd") == "cd"

    def test_no_tokens_returns_none(self):
        assert main.tokenize_query("!!! ???") is None

    def test_capped_at_16_terms(self):
        q = " ".join(f"term{i}" for i in range(30))
        assert main.tokenize_query(q).count(" OR ") == 15


class TestContentKey:
    def test_whitespace_and_case_insensitive(self):
        a = main.content_key("public  class Foo {\n}")
        b = main.content_key("public class foo {}")
        assert a == b

    def test_different_content_differs(self):
        assert main.content_key("class Foo") != main.content_key("class Bar")


class TestFilterClauses:
    def test_no_filters(self):
        assert main.filter_clauses(None, None, None) == ("", [])

    def test_repo_and_language(self):
        sql, params = main.filter_clauses("my-repo", "cs", None)
        assert "repo = %s" in sql and "language = %s" in sql
        assert params == ["my-repo", "cs"]

    def test_path_contains_escapes_like_wildcards(self):
        sql, params = main.filter_clauses(None, None, "tests\\unit_x%")
        assert "ILIKE" in sql
        assert params == [r"%tests/unit\_x\%%"]

    def test_content_type_filter(self):
        sql, params = main.filter_clauses(None, None, None, "wiki")
        assert "content_type = %s" in sql
        assert params == ["wiki"]


class TestFuseHits:
    def test_vector_only_keeps_order_and_score(self):
        hits = main.fuse_hits([make_hit("a", score=0.9), make_hit("b", text="other", score=0.5)], [], top_k=8)
        assert [h["score"] for h in hits] == [0.9, 0.5]
        assert all(h["matched_by"] == "semantic" for h in hits)

    def test_lexical_only_hit_has_no_score(self):
        hits = main.fuse_hits([], [make_hit("a")], top_k=8)
        assert hits[0]["score"] is None
        assert hits[0]["matched_by"] == "lexical"

    def test_both_legs_boost_ranking(self):
        # "c" is rank 2 in both legs; "a"/"b" lead one leg each. Two mid ranks
        # beat a single top rank under RRF.
        vector = [make_hit("a", text="aaa", score=0.9), make_hit("c", text="ccc", score=0.8)]
        lexical = [make_hit("b", text="bbb"), make_hit("c", text="ccc")]
        hits = main.fuse_hits(vector, lexical, top_k=8)
        assert hits[0]["matched_by"] == "both"
        assert hits[0]["path"] == "src/c.cs"

    def test_duplicates_collapse_into_keeper(self):
        vector = [
            make_hit("a", text="same code", score=0.9),
            make_hit("b", text="same  CODE", score=0.8, path="src/copy.cs"),
            make_hit("c", text="unique", score=0.7),
        ]
        hits = main.fuse_hits(vector, [], top_k=8)
        assert len(hits) == 2
        assert hits[0]["duplicates"] == [
            {"project": "proj", "repo": "repo", "path": "src/copy.cs", "start_line": 1}
        ]

    def test_duplicates_attach_even_past_top_k(self):
        vector = [
            make_hit("a", text="same", score=0.9),
            make_hit("b", text="unique", score=0.8),
            make_hit("c", text="same", score=0.7, path="src/late-copy.cs"),
        ]
        hits = main.fuse_hits(vector, [], top_k=2)
        assert len(hits) == 2
        assert hits[0]["duplicates"][0]["path"] == "src/late-copy.cs"

    def test_min_score_drops_semantic_only_but_keeps_lexical(self):
        vector = [make_hit("a", text="aaa", score=0.2)]
        lexical = [make_hit("b", text="bbb")]
        hits = main.fuse_hits(vector, lexical, top_k=8, min_score=0.5)
        assert [h["matched_by"] for h in hits] == ["lexical"]

    def test_compact_returns_snippet_without_text(self):
        hits = main.fuse_hits([make_hit("a", text="x" * 500, score=0.9)], [], top_k=8, compact=True)
        assert hits[0]["snippet"] == "x" * 200
        assert "text" not in hits[0]

    def test_max_chars_truncates_and_flags(self):
        hits = main.fuse_hits([make_hit("a", text="x" * 500, score=0.9)], [], top_k=8, max_chars=100)
        assert hits[0]["text"] == "x" * 100
        assert hits[0]["truncated"] is True

    def test_short_text_not_flagged_truncated(self):
        hits = main.fuse_hits([make_hit("a", text="short", score=0.9)], [], top_k=8, max_chars=100)
        assert hits[0]["text"] == "short"
        assert "truncated" not in hits[0]

    def test_internal_fields_stripped(self):
        hits = main.fuse_hits([make_hit("a", score=0.9)], [], top_k=8)
        assert "id" not in hits[0] and "rrf" not in hits[0]


class TestStitchChunks:
    def test_contiguous_chunks_join_in_order(self):
        chunks = [
            {"start_line": 3, "end_line": 4, "text": "c\nd"},
            {"start_line": 1, "end_line": 2, "text": "a\nb"},
        ]
        text, gaps = main.stitch_chunks(chunks)
        assert text == "a\nb\nc\nd"
        assert gaps == []

    def test_overlapping_lines_deduped(self):
        chunks = [
            {"start_line": 1, "end_line": 3, "text": "a\nb\nc"},
            {"start_line": 3, "end_line": 4, "text": "c\nd"},
        ]
        text, _ = main.stitch_chunks(chunks)
        assert text == "a\nb\nc\nd"

    def test_leading_gap_reported(self):
        text, gaps = main.stitch_chunks([{"start_line": 5, "end_line": 6, "text": "e\nf"}])
        assert text == "e\nf"
        assert gaps == [{"start_line": 1, "end_line": 4}]

    def test_interior_gap_reported(self):
        chunks = [
            {"start_line": 1, "end_line": 2, "text": "a\nb"},
            {"start_line": 5, "end_line": 6, "text": "e\nf"},
        ]
        text, gaps = main.stitch_chunks(chunks)
        assert text == "a\nb\ne\nf"
        assert gaps == [{"start_line": 3, "end_line": 4}]

    def test_window_clips_output(self):
        chunks = [{"start_line": 1, "end_line": 5, "text": "a\nb\nc\nd\ne"}]
        text, gaps = main.stitch_chunks(chunks, from_line=2, to_line=4)
        assert text == "b\nc\nd"
        assert gaps == []

    def test_window_past_stored_content_returns_empty(self):
        chunks = [{"start_line": 1, "end_line": 2, "text": "a\nb"}]
        assert main.stitch_chunks(chunks, from_line=50) == ("", [])

    def test_empty_returns_empty(self):
        assert main.stitch_chunks([]) == ("", [])


class TestGetConn:
    def test_unreachable_database_raises_index_not_ready(self):
        with pytest.raises(main.IndexNotReady, match="not reachable"):
            main.get_conn()
