import pytest

import main


class TestBuildWhere:
    def test_no_filters(self):
        assert main.build_where(None, None) is None

    def test_repo_only(self):
        assert main.build_where("my-repo", None) == {"repo": {"$eq": "my-repo"}}

    def test_language_only(self):
        assert main.build_where(None, "cs") == {"language": {"$eq": "cs"}}

    def test_both_filters(self):
        assert main.build_where("my-repo", "cs") == {
            "$and": [{"repo": {"$eq": "my-repo"}}, {"language": {"$eq": "cs"}}]
        }


class TestGetCollection:
    def test_missing_collection_raises_index_not_ready(self):
        # main.chroma is the stubbed HttpClient from conftest.
        main.chroma.get_collection.side_effect = ValueError("Collection code_chunks does not exist.")
        try:
            with pytest.raises(main.IndexNotReady, match="indexer completed a first run"):
                main.get_collection()
        finally:
            main.chroma.get_collection.side_effect = None
