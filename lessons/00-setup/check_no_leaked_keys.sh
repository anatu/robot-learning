#!/usr/bin/env bash
# Lesson 00 Part 4 — verify no API credentials are leaked or exposed.
# Checks: (1) no tokens in the working tree, (2) none anywhere in git
# history, (3) credential files are user-only (600), (4) no tokens
# hardcoded in shell rc files, (5) no tracked .env files.
# Exits 0 clean / 1 on any finding. Never prints secret values.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
fail=0
say() { printf '%s\n' "$*"; }

# HF tokens look like hf_<34 alnum>; wandb keys are 40 hex chars.
HF_PAT='hf_[A-Za-z0-9]{30,}'
WANDB_PAT='\b[0-9a-f]{40}\b'

say "[1/5] working tree scan"
if git grep -I -l -E "$HF_PAT|$WANDB_PAT" -- . ; then
  say "  FAIL: token-shaped string in tracked files above"; fail=1
else
  say "  ok: no token-shaped strings tracked"
fi

say "[2/5] full git history scan"
if git grep -I -l -E "$HF_PAT|$WANDB_PAT" $(git rev-list --all) -- 2>/dev/null | head -5 | grep . ; then
  say "  FAIL: token-shaped string in a past commit above"; fail=1
else
  say "  ok: history clean"
fi

say "[3/5] credential file permissions (must be 600)"
for f in "$HOME/.netrc" "$HOME/.cache/huggingface/token" "$HOME/.cache/huggingface/stored_tokens"; do
  [ -e "$f" ] || continue
  mode=$(stat -f '%Lp' "$f" 2>/dev/null || stat -c '%a' "$f")
  if [ "$mode" != "600" ]; then say "  FAIL: $f is mode $mode"; fail=1; else say "  ok: $f is 600"; fi
done

say "[4/5] shell rc files free of hardcoded tokens"
rc_fail=0
for f in "$HOME/.zshrc" "$HOME/.zshenv" "$HOME/.zprofile" "$HOME/.bashrc" "$HOME/.profile"; do
  [ -e "$f" ] || continue
  if grep -qE "$HF_PAT" "$f"; then say "  FAIL: token-shaped string in $f"; fail=1; rc_fail=1; fi
done
[ "$rc_fail" -eq 0 ] && say "  ok"

say "[5/5] no .env files tracked"
if git ls-files | grep -E '(^|/)\.env(\..*)?$' ; then
  say "  FAIL: .env file tracked above"; fail=1
else
  say "  ok"
fi

[ "$fail" -eq 0 ] && say "PASS: no credential leaks found" || say "FAIL: fix findings above"
exit "$fail"
