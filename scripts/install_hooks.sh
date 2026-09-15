#!/usr/bin/env bash
# Installs a pre-push hook that runs scripts/anon_check.sh (local blocklist required).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cat > .git/hooks/pre-push <<'H'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/anon_check.sh"
H
chmod +x .git/hooks/pre-push
echo "pre-push hook installed"
