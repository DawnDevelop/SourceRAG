#!/bin/sh
# Called by git via GIT_ASKPASS so the PAT never appears in argv/process list.
case "$1" in
  Username*) echo "pat" ;;
  Password*) echo "$AZURE_DEVOPS_PAT" ;;
esac
