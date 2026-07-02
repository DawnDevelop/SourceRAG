#!/bin/sh
set -e
python clone_repos.py
python chunk_and_embed.py
