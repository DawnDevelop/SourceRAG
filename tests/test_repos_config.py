import repos_config as rc


class TestValidRepos:
    def test_blank_value_means_repo_named_after_project(self):
        config = {"repos": {"YourInfraProject": None}}
        assert rc.valid_repos(config) == [
            {"project": "YourInfraProject", "name": "YourInfraProject", "branch": None}
        ]

    def test_string_value_is_a_single_repo_with_that_name(self):
        config = {"repos": {"YourOtherProject": "some-repo-name"}}
        assert rc.valid_repos(config) == [
            {"project": "YourOtherProject", "name": "some-repo-name", "branch": None}
        ]

    def test_list_of_strings_is_multiple_repos(self):
        config = {"repos": {"YourProject": ["service-a", "service-b"]}}
        assert rc.valid_repos(config) == [
            {"project": "YourProject", "name": "service-a", "branch": None},
            {"project": "YourProject", "name": "service-b", "branch": None},
        ]

    def test_dict_entry_carries_branch_override(self):
        config = {"repos": {"YourProject": [{"name": "service-b", "branch": "develop"}]}}
        assert rc.valid_repos(config) == [
            {"project": "YourProject", "name": "service-b", "branch": "develop"}
        ]

    def test_mixed_list_of_strings_and_dicts(self):
        config = {"repos": {"YourProject": ["service-a", {"name": "service-b", "branch": "develop"}]}}
        assert rc.valid_repos(config) == [
            {"project": "YourProject", "name": "service-a", "branch": None},
            {"project": "YourProject", "name": "service-b", "branch": "develop"},
        ]

    def test_single_dict_value_not_wrapped_in_a_list(self):
        config = {"repos": {"YourProject": {"name": "service-b", "branch": "develop"}}}
        assert rc.valid_repos(config) == [
            {"project": "YourProject", "name": "service-b", "branch": "develop"}
        ]

    def test_dict_entry_missing_name_is_skipped_not_fatal(self):
        config = {"repos": {"YourProject": [{"branch": "develop"}, "service-b"]}}
        assert rc.valid_repos(config) == [
            {"project": "YourProject", "name": "service-b", "branch": None}
        ]

    def test_unsupported_value_type_is_skipped_not_fatal(self):
        config = {"repos": {"BadProject": 42, "GoodProject": "repo"}}
        assert rc.valid_repos(config) == [
            {"project": "GoodProject", "name": "repo", "branch": None}
        ]

    def test_empty_repos_key_yields_no_repos(self):
        assert rc.valid_repos({"repos": None}) == []
        assert rc.valid_repos({"repos": {}}) == []

    def test_multiple_projects_preserve_order(self):
        config = {"repos": {"ProjA": "a", "ProjB": "b"}}
        assert [r["project"] for r in rc.valid_repos(config)] == ["ProjA", "ProjB"]


class TestShouldIndexFlags:
    def test_defaults_false_when_unset(self):
        assert rc.should_index_wikis({"azure_devops": {}}) is False
        assert rc.should_index_pull_requests({"azure_devops": {}}) is False

    def test_defaults_false_when_azure_devops_missing(self):
        assert rc.should_index_wikis({}) is False
        assert rc.should_index_pull_requests({}) is False

    def test_true_when_set(self):
        config = {"azure_devops": {"index_wikis": True, "index_pull_requests": True}}
        assert rc.should_index_wikis(config) is True
        assert rc.should_index_pull_requests(config) is True


class TestWikiRepos:
    def test_one_wiki_repo_per_distinct_project(self):
        config = {"repos": {"ProjA": ["a", "b"], "ProjB": "c"}}
        assert rc.wiki_repos(config) == [
            {"project": "ProjA", "name": "ProjA.wiki", "branch": None},
            {"project": "ProjB", "name": "ProjB.wiki", "branch": None},
        ]

    def test_no_projects_yields_no_wikis(self):
        assert rc.wiki_repos({"repos": {}}) == []
