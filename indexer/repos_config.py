"""Shared repos.yaml loading and validation for clone_repos.py and chunk_and_embed.py."""
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_project_value(project, value):
    """Turn one project's `repos` value into a list of repo specs (plain
    dicts with "name" and optionally "branch").

    Value shapes, most to least common in a large org config:
      None            -> single repo, name == project name (the common case
                          for infra/tooling projects with one repo matching
                          the project)
      "name"          -> single repo, name differs from the project
      ["a", "b"]      -> multiple repos, default branch each
      [{"name": ..., "branch": ...}, "b"]  -> mixed, per-repo branch override
    """
    if value is None:
        return [{"name": project}]
    if isinstance(value, str):
        return [{"name": value}]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [{"name": item} if isinstance(item, str) else item for item in value]
    raise ValueError(f"repos.{project}: unsupported value {value!r}")


def valid_repos(config):
    # A single malformed entry (e.g. a mistyped key while editing) shouldn't
    # take the whole ~400-repo run down -- skip it and keep going.
    repos = []
    for project, value in (config.get("repos") or {}).items():
        try:
            specs = _normalize_project_value(project, value)
        except ValueError as e:
            print(f"[config-error] {e}, skipping", flush=True)
            continue
        for spec in specs:
            if "name" not in spec:
                print(f"[config-error] repos.{project} entry is missing 'name', skipping: {spec}", flush=True)
                continue
            repos.append({"project": project, "name": spec["name"], "branch": spec.get("branch")})
    return repos


def should_index_wikis(config):
    return bool((config.get("azure_devops") or {}).get("index_wikis"))


def should_index_pull_requests(config):
    return bool((config.get("azure_devops") or {}).get("index_pull_requests"))


def wiki_repos(config):
    """One synthetic repo spec per distinct project referenced in `repos`,
    pointing at that project's wiki. Azure DevOps project wikis are
    themselves a git repo named "<project>.wiki", cloned via the exact same
    _git/ URL as any other repo -- so this reuses clone_repos.py's sync_repo
    and chunk_and_embed.py's indexing unchanged. Projects without a wiki
    simply fail to clone like any other missing repo (see clone_repos.py).
    """
    projects = sorted({r["project"] for r in valid_repos(config)})
    return [{"project": p, "name": f"{p}.wiki", "branch": None} for p in projects]
