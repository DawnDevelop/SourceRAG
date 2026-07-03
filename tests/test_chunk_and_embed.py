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


class TestChunkId:
    def test_deterministic(self):
        a = cae.chunk_id("proj", "repo", "src/a.py", 0)
        b = cae.chunk_id("proj", "repo", "src/a.py", 0)
        assert a == b

    def test_distinct_per_index(self):
        assert cae.chunk_id("proj", "repo", "src/a.py", 0) != cae.chunk_id("proj", "repo", "src/a.py", 1)

    def test_distinct_per_repo(self):
        assert cae.chunk_id("proj", "repo1", "src/a.py", 0) != cae.chunk_id("proj", "repo2", "src/a.py", 0)


class TestConfigFingerprint:
    def test_stable_across_calls(self):
        assert cae.config_fingerprint() == cae.config_fingerprint()

    def test_changes_with_chunk_size(self, monkeypatch):
        before = cae.config_fingerprint()
        monkeypatch.setattr(cae, "CHUNK_SIZE", cae.CHUNK_SIZE + 1)
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
