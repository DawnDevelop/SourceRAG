import pull_requests as pr


class FakeResponse:
    def __init__(self, value):
        self._value = value

    def raise_for_status(self):
        pass

    def json(self):
        return {"value": self._value}


class FakeSession:
    """Records requested $skip values and serves pre-built PR list pages in
    order; work-item lookups are routed by URL and served from `work_items`
    (defaulting to none linked for any PR not mentioned there)."""

    def __init__(self, pages, work_items=None):
        self.pages = list(pages)
        self.work_items = work_items or {}
        self.requested_skips = []

    def get(self, url, params, timeout):
        if url.endswith("/workitems"):
            pr_id = int(url.rstrip("/").split("/")[-2])
            return FakeResponse([{"id": str(i)} for i in self.work_items.get(pr_id, [])])
        self.requested_skips.append(params["$skip"])
        page = self.pages.pop(0) if self.pages else []
        return FakeResponse(page)


def make_pr(pr_id, title="t", description="d", created="2026-01-01T00:00:00Z"):
    return {"pullRequestId": pr_id, "title": title, "description": description, "creationDate": created}


class TestFetchNewPullRequests:
    def test_first_run_walks_entire_history_oldest_first(self):
        session = FakeSession([[make_pr(3), make_pr(2)], [make_pr(1)], []])
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo", since_id=0)
        assert [p["id"] for p in prs] == [1, 2, 3]

    def test_stops_once_a_page_hits_the_cursor(self):
        # Newest-first page: id 5 is new, id 3 is already indexed -- should
        # stop after this page without requesting a second one.
        session = FakeSession([[make_pr(5), make_pr(3)], [make_pr(2)]])
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo", since_id=3)
        assert [p["id"] for p in prs] == [5]
        assert session.requested_skips == [0]

    def test_no_new_prs_returns_empty(self):
        session = FakeSession([[make_pr(3), make_pr(2)]])
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo", since_id=3)
        assert prs == []

    def test_empty_first_page_returns_empty(self):
        session = FakeSession([[]])
        assert pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo") == []

    def test_maps_fields(self):
        session = FakeSession([[make_pr(1, title="Add X", description="Body", created="2026-02-02T00:00:00Z")]])
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo")
        assert prs == [{"id": 1, "title": "Add X", "description": "Body",
                        "created_date": "2026-02-02T00:00:00Z", "work_item_ids": []}]

    def test_missing_title_and_description_default_to_empty_string(self):
        session = FakeSession([[{"pullRequestId": 1, "creationDate": "2026-01-01T00:00:00Z"}]])
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo")
        assert prs[0]["title"] == ""
        assert prs[0]["description"] == ""

    def test_linked_work_item_ids_are_attached_and_sorted(self):
        session = FakeSession([[make_pr(1)]], work_items={1: [42, 7]})
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo")
        assert prs[0]["work_item_ids"] == [7, 42]

    def test_no_linked_work_items_is_empty_list(self):
        session = FakeSession([[make_pr(1)]])
        prs = pr.fetch_new_pull_requests(session, "https://dev.azure.com/org", "Proj", "repo")
        assert prs[0]["work_item_ids"] == []


class TestFetchLinkedWorkItemIds:
    def test_returns_ids_as_ints(self):
        session = FakeSession([], work_items={5: [100, 200]})
        ids = pr.fetch_linked_work_item_ids(session, "https://dev.azure.com/org", "Proj", "repo", 5)
        assert ids == [100, 200]

    def test_no_links_returns_empty(self):
        session = FakeSession([])
        assert pr.fetch_linked_work_item_ids(session, "https://dev.azure.com/org", "Proj", "repo", 5) == []


class TestMakeSession:
    def test_auth_uses_empty_username_and_pat_as_password(self):
        session = pr.make_session("my-pat")
        assert session.auth == ("", "my-pat")
