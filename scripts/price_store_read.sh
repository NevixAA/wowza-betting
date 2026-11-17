#!/usr/bin/env bash
# Read prices from the PRIVATE store instead of the public files (owner decision 2026-11-17).
#
#   scripts/price_store_read.sh <repo> <branch>
#
# Clones NevixAA/wowza-exec@af-prices-live READ-ONLY (deploy key in PRICE_STORE_RO_KEY), sparse, into
# $RUNNER_TEMP, then scripts/price_store_hydrate.py --prune puts the store's class A / B / O files at
# their public paths and removes any selected workspace file the store does not hold. The run then
# reads ONLY store bytes. Afterwards `git status` must be clean: the store bytes are byte-identical to
# the public commit, or the job stops here, before any training or scoring.
#
# Workflows call this only when vars.PRICE_STORE_READ == '1' (default off = public files, today's
# behaviour). Rollback is one line: set PRICE_STORE_READ to 0. Logs carry counts only.
set -euo pipefail

REPO_NAME="${1:?repo}"
BRANCH_NAME="${2:?branch}"
STORE="${RUNNER_TEMP:-/tmp}/price_store_ro"
KEY="${RUNNER_TEMP:-/tmp}/price_store_ro_key"
KNOWN="${RUNNER_TEMP:-/tmp}/price_store_ro_known_hosts"
PREFIX="$REPO_NAME/${BRANCH_NAME//\//__}"

if [ -z "${PRICE_STORE_RO_KEY:-}" ]; then
  echo "::error title=price store::PRICE_STORE_READ is on but secret PRICE_STORE_RO_KEY is not set"; exit 1
fi
( umask 077; printf '%s\n' "$PRICE_STORE_RO_KEY" > "$KEY" )
# GitHub's published ed25519 host key (docs.github.com, "GitHub's SSH key fingerprints").
echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" > "$KNOWN"
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN"
rm -rf "$STORE"
git clone --quiet --depth 1 --filter=blob:none --sparse --branch af-prices-live \
  "${PRICE_STORE_URL:-git@github.com:NevixAA/wowza-exec.git}" "$STORE"
git -C "$STORE" sparse-checkout set --no-cone "/.price_store" "/_sync/" "/$PREFIX/"
rm -f "$KEY"
echo "[price_store] store $(git -C "$STORE" rev-parse --short HEAD) opened read-only for $PREFIX"

python3 scripts/price_store_hydrate.py --repo "$REPO_NAME" --branch "$BRANCH_NAME" --workspace . \
  --store-dir "$STORE" --prune --verify --report "${RUNNER_TEMP:-/tmp}/price_store_hydrate.json"

changed=$(git status --porcelain | wc -l)
if [ "$changed" != "0" ]; then
  echo "::error title=price store::$changed file(s) differ between the private store and this commit - stopping before any work (rollback: set PRICE_STORE_READ to 0)"
  git status --porcelain | awk '{print $1}' | sort | uniq -c
  exit 1
fi
echo "PRICE_STORE_DIR=$STORE" >> "${GITHUB_ENV:-/dev/null}"
echo "[price_store] reading prices from the private store: workspace byte-identical to the public commit"
