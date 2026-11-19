#!/usr/bin/env bash
# Private corners/cards odds store (owner 2026-11-16, audit gap G1).
#
#   scripts/side_odds_store.sh open             clone the store, export SIDE_ODDS_DIR via $GITHUB_ENV
#   scripts/side_odds_store.sh push <producer>  commit + push this run's appended rows
#
# The quotes are API-Football prices. Its terms grant no licence to publish them, and this repo is
# PUBLIC, so they live ONLY in the private repo NevixAA/wowza-exec, orphan branch `af-side-odds`,
# cloned outside the workspace ($RUNNER_TEMP). Access is a write deploy key on wowza-exec held in the
# secret SIDE_ODDS_STORE_KEY. Nothing here prints a file's contents: counts and status only.
#
# OPEN never fails the job: without the store the capture runs exactly as before and keeps no
# corners/cards quotes. PUSH fails the job when the store was requested but could not be opened or
# pushed, so a lost day is visible (it runs after the public commit, which it cannot affect).
# This file is identical in wowza-betting and wowzaV9-Pro.
set -uo pipefail

REPO="git@github.com:NevixAA/wowza-exec.git"
BRANCH="af-side-odds"
DIR="${RUNNER_TEMP:-/tmp}/side_odds_store"
KEY="${RUNNER_TEMP:-/tmp}/side_odds_key"
KNOWN="${RUNNER_TEMP:-/tmp}/side_odds_known_hosts"
FLAG="${RUNNER_TEMP:-/tmp}/side_odds_open_failed"

_ssh() {
  # GitHub's published ed25519 host key (docs.github.com, "GitHub's SSH key fingerprints").
  echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" > "$KNOWN"
  export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN"
}

case "${1:-}" in
  open)
    rm -f "$FLAG"
    if [ -z "${SIDE_ODDS_STORE_KEY:-}" ]; then
      echo "::warning title=side odds::SIDE_ODDS_STORE_KEY is not set: corners/cards odds are not kept this run"
      touch "$FLAG"; exit 0
    fi
    ( umask 077; printf '%s\n' "$SIDE_ODDS_STORE_KEY" > "$KEY" )
    _ssh
    rm -rf "$DIR"
    if ! git clone --quiet --depth 1 --branch "$BRANCH" "$REPO" "$DIR" 2>/dev/null; then
      echo "::warning title=side odds::private store clone failed: corners/cards odds are not kept this run"
      touch "$FLAG"; exit 0
    fi
    if [ ! -f "$DIR/.side_odds_store" ]; then
      echo "::warning title=side odds::store marker missing: corners/cards odds are not kept this run"
      rm -rf "$DIR"; touch "$FLAG"; exit 0
    fi
    echo "SIDE_ODDS_DIR=$DIR" >> "$GITHUB_ENV"
    echo "[side_odds] private store open ($(find "$DIR/data" -name '*.csv' 2>/dev/null | wc -l) day file(s))"
    ;;
  push)
    PRODUCER="${2:?producer}"
    if [ -f "$FLAG" ] || [ ! -d "$DIR/.git" ]; then
      rm -f "$KEY"
      echo "::error title=side odds::the private store was not opened, so this run kept no corners/cards odds"
      exit 1
    fi
    _ssh
    cd "$DIR" || exit 1
    git config user.name  "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add data 2>/dev/null || true
    if git diff --staged --quiet; then
      echo "[side_odds] nothing new for $PRODUCER"; rm -f "$KEY"; exit 0
    fi
    ADDED=$(git diff --staged --numstat | awk '{s += $1} END {print s + 0}')
    FILES=$(git diff --staged --name-only | wc -l)
    git commit --quiet -m "side-odds: $PRODUCER $(date -u '+%Y-%m-%d %H:%M UTC') (+$ADDED lines, $FILES file(s))"
    ok=0
    for i in 1 2 3 4 5; do
      if git pull --quiet --rebase origin "$BRANCH" 2>/dev/null && git push --quiet origin "HEAD:$BRANCH" 2>/dev/null; then
        ok=1; break
      fi
      echo "[side_odds] push race - retry $i"; sleep $((RANDOM % 5 + 3))
    done
    rm -f "$KEY"
    [ "$ok" = "1" ] || { echo "::error title=side odds::push to the private store failed after 5 tries"; exit 1; }
    echo "[side_odds] pushed $PRODUCER: +$ADDED line(s) in $FILES file(s) -> private store"
    ;;
  *)
    echo "usage: $0 open | push <producer>"; exit 2 ;;
esac
