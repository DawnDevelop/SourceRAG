"""Azure DevOps pull request fetching -- title + description per PR, indexed
alongside code so search can surface the "why" behind a change, not just the
diff (see chunk_and_embed.py's embed_pull_requests/index_pull_requests).

Separate from clone_repos.py's git-based sync: PRs aren't git objects, so
they're fetched via the REST API instead of a clone/pull.
"""
from urllib.parse import quote

import requests

API_VERSION = "7.1"
PAGE_SIZE = 100


def make_session(pat):
    session = requests.Session()
    # ADO's REST API accepts the PAT as the password half of HTTP Basic auth
    # with an empty username -- same convention as GIT_ASKPASS in git-askpass.sh.
    session.auth = ("", pat)
    return session


def _pull_requests_url(org_url, project, repo):
    return f"{org_url.rstrip('/')}/{quote(project)}/_apis/git/repositories/{quote(repo)}/pullrequests"


def _work_items_url(org_url, project, repo, pr_id):
    return (f"{org_url.rstrip('/')}/{quote(project)}/_apis/git/repositories/{quote(repo)}"
            f"/pullRequests/{pr_id}/workitems")


def fetch_linked_work_item_ids(session, org_url, project, repo, pr_id):
    """Work item IDs linked to a PR -- just the numbers, not their content.
    This is a Git API endpoint (returns ResourceRef stubs: id + url), not the
    Work Item Tracking API, so it needs no PAT scope beyond Code (Read) and
    never touches a work item's title/description/comments (where personal
    data like reporter/customer names would actually live).
    """
    resp = session.get(_work_items_url(org_url, project, repo, pr_id),
                       params={"api-version": API_VERSION}, timeout=30)
    resp.raise_for_status()
    return sorted(int(item["id"]) for item in resp.json().get("value", []))


def fetch_new_pull_requests(session, org_url, project, repo, since_id=0):
    """All PRs (any status) for `repo` with id > since_id, oldest-first.

    The list API has no server-side id filter and returns newest-first, so
    pagination stops as soon as a page contains an id <= since_id -- on a
    repo with nothing new that's a single page. `since_id=0` (first run for
    this repo) walks its entire PR history.
    """
    url = _pull_requests_url(org_url, project, repo)
    collected = []
    skip = 0
    while True:
        resp = session.get(url, params={
            "api-version": API_VERSION,
            "searchCriteria.status": "all",
            "$top": PAGE_SIZE,
            "$skip": skip,
        }, timeout=30)
        resp.raise_for_status()
        page = resp.json().get("value", [])
        if not page:
            break

        new_on_page = [pr for pr in page if pr["pullRequestId"] > since_id]
        collected.extend(new_on_page)
        if len(new_on_page) < len(page):
            break  # this page ran into already-indexed PRs -- older pages are all old too
        skip += PAGE_SIZE

    mapped = [
        {
            "id": pr["pullRequestId"],
            "title": pr.get("title") or "",
            "description": pr.get("description") or "",
            "created_date": pr.get("creationDate") or "",
            "work_item_ids": fetch_linked_work_item_ids(session, org_url, project, repo, pr["pullRequestId"]),
        }
        for pr in collected
    ]
    return sorted(mapped, key=lambda pr: pr["id"])
