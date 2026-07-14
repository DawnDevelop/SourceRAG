import subprocess

import pytest

import chunk_and_embed as cae


class TestPathIsSkipped:
    def test_normal_source_file(self):
        assert not cae.path_is_skipped("src/services/auth.py")

    def test_skip_dir_anywhere_in_path(self):
        assert cae.path_is_skipped("frontend/node_modules/lodash/index.js")

    def test_skip_dir_name_as_filename_is_not_skipped(self):
        # Only *directories* named e.g. "bin" are skipped, not a file
        # that happens to carry the same name.
        assert not cae.path_is_skipped("scripts/bin")

    def test_skip_extension(self):
        assert cae.path_is_skipped("assets/logo.png")

    def test_skip_extension_case_insensitive(self):
        assert cae.path_is_skipped("assets/LOGO.PNG")

    def test_test_code_dirs_are_indexed(self):
        assert not cae.path_is_skipped("tests/test_auth.py")

    def test_locale_dirs_are_skipped(self):
        assert cae.path_is_skipped("public/locales/pl/translation.json")

    def test_minified_suffix_is_skipped(self):
        assert cae.path_is_skipped("js/model-viewer.min.js")

    def test_lockfiles_are_skipped(self):
        assert cae.path_is_skipped("frontend/package-lock.json")
        assert cae.path_is_skipped("Gemfile.lock")

    def test_source_maps_are_skipped(self):
        assert cae.path_is_skipped("dist-out/bundle.js.map")


class TestChunkId:
    def test_deterministic(self):
        a = cae.chunk_id("proj", "repo", "src/a.py", 0)
        b = cae.chunk_id("proj", "repo", "src/a.py", 0)
        assert a == b

    def test_distinct_per_index(self):
        assert cae.chunk_id("proj", "repo", "src/a.py", 0) != cae.chunk_id("proj", "repo", "src/a.py", 1)

    def test_distinct_per_repo(self):
        assert cae.chunk_id("proj", "repo1", "src/a.py", 0) != cae.chunk_id("proj", "repo2", "src/a.py", 0)


class TestContentHash:
    def test_whitespace_and_case_insensitive(self):
        a = cae.content_hash("public  class Foo {\n}")
        b = cae.content_hash("public class foo {}")
        assert a == b

    def test_different_content_differs(self):
        assert cae.content_hash("class Foo") != cae.content_hash("class Bar")


class TestConfigFingerprint:
    def test_stable_across_calls(self):
        assert cae.config_fingerprint() == cae.config_fingerprint()

    def test_changes_with_OUTPUT_CHUNK_SIZE(self, monkeypatch):
        before = cae.config_fingerprint()
        monkeypatch.setattr(cae, "OUTPUT_CHUNK_SIZE", cae.OUTPUT_CHUNK_SIZE + 1)
        assert cae.config_fingerprint() != before


class TestLineOffsets:
    TEXT = "line one\nline two\nline three\n"

    def test_offset_to_line(self):
        offsets = cae.line_offsets(self.TEXT)
        assert cae.offset_to_line(offsets, 0) == 1
        assert cae.offset_to_line(offsets, self.TEXT.index("two")) == 2
        assert cae.offset_to_line(offsets, self.TEXT.index("three")) == 3

    def test_line_boundaries(self):
        offsets = cae.line_offsets(self.TEXT)
        # Last char of line 1 (the \n) is still line 1; first char after is line 2.
        assert cae.offset_to_line(offsets, len("line one")) == 1
        assert cae.offset_to_line(offsets, len("line one\n")) == 2

    def test_no_trailing_newline(self):
        offsets = cae.line_offsets("a\nb")
        assert cae.offset_to_line(offsets, 2) == 2


class TestParseNameStatus:
    def test_add_modify_delete(self):
        out = "A\tnew.py\nM\tchanged.py\nD\tgone.py\n"
        assert cae.parse_name_status(out) == [("A", "new.py"), ("M", "changed.py"), ("D", "gone.py")]

    def test_rename_resolves_to_delete_plus_add(self):
        out = "R100\told/name.py\tnew/name.py\n"
        assert cae.parse_name_status(out) == [("D", "old/name.py"), ("A", "new/name.py")]

    def test_typechange_treated_as_modify(self):
        assert cae.parse_name_status("T\tlink.py\n") == [("M", "link.py")]

    def test_blank_lines_ignored(self):
        assert cae.parse_name_status("\nA\ta.py\n\n") == [("A", "a.py")]


@pytest.fixture
def git_repo(tmp_path):
    def run(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@test.invalid", *args],
            check=True, capture_output=True,
        )

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    return tmp_path, run


class TestGitDiffStatus:
    def test_real_repo_add_modify_delete_rename(self, git_repo):
        repo, run = git_repo
        # Enough content that git's rename detection (-M) recognizes the move.
        (repo / "keep.py").write_text("def keep():\n    return 1\n" * 20)
        (repo / "gone.py").write_text("def gone():\n    return 2\n")
        (repo / "moved.py").write_text("def moved():\n    return 3\n" * 20)
        run("add", "-A")
        run("commit", "-q", "-m", "first")
        old = cae.get_head_commit(str(repo))

        (repo / "keep.py").write_text("def keep():\n    return 99\n" * 20)
        (repo / "gone.py").unlink()
        (repo / "moved.py").rename(repo / "renamed.py")
        (repo / "new.py").write_text("def new():\n    return 4\n")
        run("add", "-A")
        run("commit", "-q", "-m", "second")
        new = cae.get_head_commit(str(repo))

        changes = set(cae.git_diff_status(str(repo), old, new))
        assert changes == {
            ("M", "keep.py"),
            ("D", "gone.py"),
            ("D", "moved.py"),
            ("A", "renamed.py"),
            ("A", "new.py"),
        }


class TestIsMinified:
    def test_normal_code_is_not_minified(self):
        text = "def foo():\n    return 1\n" * 200
        assert not cae.is_minified(text)

    def test_single_huge_line_is_minified(self):
        assert cae.is_minified("var a=1;" * 1000)

    def test_short_single_line_file_is_exempt(self):
        assert not cae.is_minified("x" * 500)


class TestIsImportDominated:
    def test_csharp_using_block(self):
        chunk = "\n".join(f"using System.Collections.Generic{i};" for i in range(10))
        assert cae.is_import_dominated(chunk)

    def test_python_import_block(self):
        chunk = "import os\nimport sys\nfrom pathlib import Path\nimport json\n"
        assert cae.is_import_dominated(chunk)

    def test_real_code_is_kept(self):
        chunk = "using System;\n\npublic class Foo\n{\n    public int Bar() => 1;\n}\n"
        assert not cae.is_import_dominated(chunk)

    def test_csharp_using_statement_is_not_an_import(self):
        # `using var stream = ...` is a resource statement, not a directive.
        chunk = "using var a = Open();\nusing var b = Open();\nusing var c = Open();\n"
        assert not cae.is_import_dominated(chunk)

    def test_tiny_chunks_are_kept(self):
        assert not cae.is_import_dominated("import os\nimport sys")


class TestIterChunkRows:
    def make_file(self, tmp_path, name, content, encoding="utf-8"):
        path = tmp_path / name
        path.write_text(content, encoding=encoding)
        return str(path), path.suffix.lower()

    def test_bom_only_file_yields_nothing(self, tmp_path):
        spec = self.make_file(tmp_path, "empty.cs", "﻿")
        rows = list(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec]))
        assert rows == []

    def test_bom_is_stripped_from_content(self, tmp_path):
        spec = self.make_file(tmp_path, "code.cs", "﻿public class Foo { public int Bar() => 1; }")
        rows = list(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec]))
        assert rows and not rows[0][2].startswith("﻿")

    def test_nul_bytes_are_stripped_from_content(self, tmp_path):
        spec = self.make_file(tmp_path, "code.cs", "public class Foo {\x00 public int Bar() => 1; }")
        rows = list(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec]))
        assert rows and "\x00" not in rows[0][2]

    def test_minified_file_yields_nothing(self, tmp_path):
        spec = self.make_file(tmp_path, "bundle.js", "var a=1;" * 1000)
        rows = list(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec]))
        assert rows == []

    def test_embed_text_carries_path_header_but_stored_text_does_not(self, tmp_path):
        content = "public class Foo { public int Bar() => 1; }"
        spec = self.make_file(tmp_path, "code.cs", content)
        cid, payload, text, embed_text = next(iter(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec])))
        assert embed_text.startswith("// proj/repo/code.cs\n")
        assert text == content
        assert payload["language"] == "cs"
        assert cid == cae.chunk_id("proj", "repo", "code.cs", 0)

    def test_commit_fields_are_blank_placeholders(self, tmp_path):
        # commit_hash/commit_date are backfilled by embed_files() from a background
        # commit-map lookup (see TestEmbedFilesCommitMap) -- iter_chunk_rows itself
        # never resolves them, so it always yields blank placeholders.
        spec = self.make_file(tmp_path, "code.cs", "public class Foo { public int Bar() => 1; }")
        _, payload, _, _ = next(iter(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec])))
        assert payload["commit_hash"] == ""
        assert payload["commit_date"] == ""

    def test_content_type_defaults_to_code(self, tmp_path):
        spec = self.make_file(tmp_path, "code.cs", "public class Foo { public int Bar() => 1; }")
        _, payload, _, _ = next(iter(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec])))
        assert payload["content_type"] == "code"

    def test_content_type_is_threaded_through(self, tmp_path):
        spec = self.make_file(tmp_path, "page.md", "# A wiki page\n\nSome content.")
        _, payload, _, _ = next(iter(
            cae.iter_chunk_rows("proj", "repo.wiki", str(tmp_path), [spec], content_type="wiki")
        ))
        assert payload["content_type"] == "wiki"

    def test_content_hash_matches_stored_text(self, tmp_path):
        content = "public class Foo { public int Bar() => 1; }"
        spec = self.make_file(tmp_path, "code.cs", content)
        _, payload, text, _ = next(iter(cae.iter_chunk_rows("proj", "repo", str(tmp_path), [spec])))
        assert payload["content_hash"] == cae.content_hash(text)


class TestChunkPayloads:
    def test_default_content_type_is_code(self):
        _, payload, _, _ = next(iter(
            cae.chunk_payloads("proj", "repo", "src/a.py", "print(1)", ".py", 8)
        ))
        assert payload["content_type"] == "code"

    def test_commit_date_override_is_used_verbatim(self):
        _, payload, _, _ = next(iter(
            cae.chunk_payloads("proj", "repo", "_pull_requests/1.md", "# Title\n\nBody", ".md", 13,
                               content_type="pr", commit_date="2026-01-01T00:00:00Z")
        ))
        assert payload["commit_date"] == "2026-01-01T00:00:00Z"
        assert payload["commit_hash"] == ""
        assert payload["content_type"] == "pr"


class TestEmbedFilesCommitMap:
    def make_file(self, tmp_path, name, content):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return str(path), path.suffix.lower()

    def test_backfills_commit_hash_and_date(self, tmp_path, monkeypatch):
        spec = self.make_file(tmp_path, "code.cs", "public class Foo { public int Bar() => 1; }")
        monkeypatch.setattr(
            cae, "get_last_commit_map",
            lambda repo_root, rel_paths: {"code.cs": ("deadbeef", "2026-01-01T00:00:00Z")},
        )
        monkeypatch.setattr(cae, "embed_batch", lambda texts: [[0.0]] * len(texts))
        written = []
        monkeypatch.setattr(cae, "add_chunks", lambda ids, payloads, texts, vectors: written.extend(payloads))

        cae.embed_files("proj", "repo", str(tmp_path), [spec])

        assert written[0]["commit_hash"] == "deadbeef"
        assert written[0]["commit_date"] == "2026-01-01T00:00:00Z"

    def test_commit_map_failure_propagates(self, tmp_path, monkeypatch):
        spec = self.make_file(tmp_path, "code.cs", "public class Foo { public int Bar() => 1; }")

        def boom(repo_root, rel_paths):
            raise RuntimeError("git log failed")

        monkeypatch.setattr(cae, "get_last_commit_map", boom)
        monkeypatch.setattr(cae, "embed_batch", lambda texts: [[0.0]] * len(texts))
        monkeypatch.setattr(cae, "add_chunks", lambda *a, **k: None)

        with pytest.raises(RuntimeError, match="git log failed"):
            cae.embed_files("proj", "repo", str(tmp_path), [spec])


class TestVectorLiteral:
    def test_format(self):
        assert cae.vector_literal([0.5, -1.0, 2.25]) == "[0.5,-1.0,2.25]"


class TestEmbedPullRequests:
    def test_empty_list_short_circuits(self):
        assert cae.embed_pull_requests("proj", "repo", []) == 0

    def test_pr_becomes_a_chunk_with_title_and_description(self, monkeypatch):
        monkeypatch.setattr(cae, "embed_batch", lambda texts: [[0.0]] * len(texts))
        written = []
        monkeypatch.setattr(cae, "add_chunks", lambda ids, payloads, texts, vectors: written.extend(
            zip(payloads, texts)
        ))

        prs = [{"id": 42, "title": "Add retry policy", "description": "Wraps calls in Polly.",
                "created_date": "2026-01-01T00:00:00Z"}]
        chunk_count = cae.embed_pull_requests("proj", "repo", prs)

        assert chunk_count == 1
        payload, text = written[0]
        assert payload["content_type"] == "pr"
        assert payload["path"] == "_pull_requests/42.md"
        assert payload["commit_date"] == "2026-01-01T00:00:00Z"
        assert "Add retry policy" in text
        assert "Wraps calls in Polly." in text

    def test_blank_description_does_not_crash(self, monkeypatch):
        monkeypatch.setattr(cae, "embed_batch", lambda texts: [[0.0]] * len(texts))
        written = []
        monkeypatch.setattr(cae, "add_chunks", lambda ids, payloads, texts, vectors: written.extend(payloads))

        prs = [{"id": 1, "title": "No description PR", "description": "", "created_date": ""}]
        cae.embed_pull_requests("proj", "repo", prs)

        assert written[0]["content_type"] == "pr"

    def test_linked_work_item_ids_are_appended_as_bare_numbers(self, monkeypatch):
        monkeypatch.setattr(cae, "embed_batch", lambda texts: [[0.0]] * len(texts))
        written = []
        monkeypatch.setattr(cae, "add_chunks", lambda ids, payloads, texts, vectors: written.extend(texts))

        # Already sorted, as fetch_linked_work_item_ids() would provide it --
        # embed_pull_requests renders in the order it's given, it doesn't re-sort.
        prs = [{"id": 1, "title": "Fix bug", "description": "Root cause was X.",
                "created_date": "", "work_item_ids": [7, 42]}]
        cae.embed_pull_requests("proj", "repo", prs)

        assert "Linked work items: #7, #42" in written[0]

    def test_no_linked_work_items_adds_no_extra_text(self, monkeypatch):
        monkeypatch.setattr(cae, "embed_batch", lambda texts: [[0.0]] * len(texts))
        written = []
        monkeypatch.setattr(cae, "add_chunks", lambda ids, payloads, texts, vectors: written.extend(texts))

        prs = [{"id": 1, "title": "Fix bug", "description": "Root cause was X.",
                "created_date": "", "work_item_ids": []}]
        cae.embed_pull_requests("proj", "repo", prs)

        assert "Linked work items" not in written[0]


class TestIndexPullRequests:
    def test_fetches_since_last_cursor_and_updates_state(self, monkeypatch):
        captured_since_id = []

        def fake_fetch(session, org_url, project, repo, since_id):
            captured_since_id.append(since_id)
            return [{"id": 5, "title": "t", "description": "d", "created_date": "2026-01-01T00:00:00Z"}]

        monkeypatch.setattr(cae.pull_requests, "fetch_new_pull_requests", fake_fetch)
        monkeypatch.setattr(cae, "embed_pull_requests", lambda project, name, prs: len(prs))
        monkeypatch.setattr(cae, "_write_state_to_disk", lambda state: None)

        state = {"pull_requests": {"proj/repo": {"last_pr_id": 3}}}
        cae.index_pull_requests(object(), "https://dev.azure.com/org", "proj", "repo", state)

        assert captured_since_id == [3]
        assert state["pull_requests"]["proj/repo"]["last_pr_id"] == 5

    def test_no_new_prs_leaves_state_untouched(self, monkeypatch):
        monkeypatch.setattr(cae.pull_requests, "fetch_new_pull_requests", lambda *a, **k: [])
        monkeypatch.setattr(cae, "_write_state_to_disk", lambda state: None)

        state = {"pull_requests": {"proj/repo": {"last_pr_id": 3}}}
        cae.index_pull_requests(object(), "https://dev.azure.com/org", "proj", "repo", state)

        assert state["pull_requests"]["proj/repo"]["last_pr_id"] == 3
