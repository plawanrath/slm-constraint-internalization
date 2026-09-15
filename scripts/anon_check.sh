#!/usr/bin/env bash
# Fails if any tracked file or any commit in history contains an identifying term.
# Run before every push and before building the artifact bundle.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
BLOCKLIST_FILE="${SCI_ANON_BLOCKLIST:-scripts/anon_blocklist.txt}"
if [ ! -f "$BLOCKLIST_FILE" ]; then
  echo "anon_check: blocklist $BLOCKLIST_FILE missing (it is gitignored; create it locally)" >&2
  exit 2
fi
PATTERN="$(grep -vE '^\s*(#|$)' "$BLOCKLIST_FILE" | paste -sd'|' -)"
status=0
echo "== tracked files =="
if git grep -nIiE "$PATTERN" -- . 2>/dev/null | grep .; then status=1; fi
echo "== git history (patches) =="
if git log -p --all 2>/dev/null | grep -nIiE "$PATTERN" | head -20 | grep .; then status=1; fi
echo "== git authorship =="
if git log --all --format='%an <%ae>' 2>/dev/null | sort -u | grep -viE '^(anon|anonymous) ' | grep .; then
  echo "note: commits carry a real author identity; rewrite before release" ; fi
if [ $status -ne 0 ]; then echo "anon_check: FAIL" >&2; exit 1; fi
echo "anon_check: OK"
