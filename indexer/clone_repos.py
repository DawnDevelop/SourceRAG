import os
import subprocess
from urllib.parse import quote

import yaml

CONFIG_PATH = os.environ.get("REPOS_CONFIG", "/config/repos.yaml")
DATA_DIR = os.environ.get("REPO_DATA_DIR", "/data/repos")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_url(org_url, project, name):
    return f"{org_url.rstrip('/')}/{quote(project)}/_git/{quote(name)}"


def sync_repo(org_url, project, name, branch=None):
    url = repo_url(org_url, project, name)
    dest = os.path.join(DATA_DIR, project, name)

    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"[sync] pulling {project}/{name}")
        if branch:
            subprocess.run(["git", "-C", dest, "fetch", "origin", branch], check=True)
            subprocess.run(["git", "-C", dest, "checkout", branch], check=True)
            subprocess.run(["git", "-C", dest, "reset", "--hard", f"origin/{branch}"], check=True)
        else:
            # No branch pinned in config -- fast-forward whatever branch was
            # checked out at clone time (the repo's default branch).
            subprocess.run(["git", "-C", dest, "pull", "--ff-only"], check=True)
    else:
        print(f"[sync] cloning {project}/{name}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cmd = ["git", "clone", "--single-branch"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [url, dest]
        subprocess.run(cmd, check=True)


def main():
    config = load_config()
    org_url = config["azure_devops"]["organization"]
    for repo in config["repos"]:
        sync_repo(org_url, repo["project"], repo["name"], repo.get("branch"))


if __name__ == "__main__":
    main()
