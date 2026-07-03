"""Shared repos.yaml loading and validation for clone_repos.py and chunk_and_embed.py."""
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def valid_repos(config):
    # A single malformed entry (e.g. a "name:" line misplaced during editing)
    # shouldn't take the whole ~400-repo run down -- skip it and keep going.
    repos = []
    for i, r in enumerate(config["repos"]):
        if "project" not in r or "name" not in r:
            print(f"[config-error] repos.yaml entry #{i} is missing 'project' or 'name', skipping: {r}", flush=True)
            continue
        repos.append(r)
    return repos
