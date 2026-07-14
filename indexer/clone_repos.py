import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

from repos_config import load_config, valid_repos, wiki_repos, should_index_wikis

CONFIG_PATH = os.environ.get("REPOS_CONFIG", "/config/repos.yaml")
DATA_DIR = os.environ.get("REPO_DATA_DIR", "/data/repos")
CLONE_CONCURRENCY = int(os.environ.get("CLONE_CONCURRENCY", "8"))


def repo_url(org_url, project, name):
    return f"{org_url.rstrip('/')}/{quote(project)}/_git/{quote(name)}"


def sync_repo(org_url, project, name, branch=None):
    url = repo_url(org_url, project, name)
    dest = os.path.join(DATA_DIR, project, name)

    # Captured (not streamed) so N parallel git processes don't interleave
    # garbled progress output -- only shown on failure, via CalledProcessError.
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"[sync] pulling {project}/{name}")
        if branch:
            subprocess.run(["git", "-C", dest, "fetch", "origin", branch], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", dest, "checkout", branch], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", dest, "reset", "--hard", f"origin/{branch}"], check=True, capture_output=True, text=True)
        else:
            # No branch pinned in config -- fast-forward whatever branch was
            # checked out at clone time (the repo's default branch).
            subprocess.run(["git", "-C", dest, "pull", "--ff-only"], check=True, capture_output=True, text=True)
    else:
        print(f"[sync] cloning {project}/{name}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cmd = ["git", "clone", "--single-branch"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [url, dest]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    config = load_config(CONFIG_PATH)
    org_url = config["azure_devops"]["organization"]
    repos = valid_repos(config)
    if should_index_wikis(config):
        # A project with no wiki simply has no ".wiki" git repo to clone --
        # that shows up below as an ordinary clone failure, not a crash.
        repos = repos + wiki_repos(config)

    failures = []
    with ThreadPoolExecutor(max_workers=CLONE_CONCURRENCY) as pool:
        futures = {
            pool.submit(sync_repo, org_url, r["project"], r["name"], r.get("branch")): r
            for r in repos
        }
        for future in as_completed(futures):
            r = futures[future]
            label = f"{r['project']}/{r['name']}"
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                failures.append(label)
                last_line = (e.stderr or "").strip().splitlines()[-1:] or [""]
                print(f"[fail] {label}: {last_line[0]}")

    print(f"[sync] done: {len(repos) - len(failures)}/{len(repos)} ok, {len(failures)} failed")
    if failures:
        print("[sync] failed repos: " + ", ".join(failures))


if __name__ == "__main__":
    main()
