#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  REPO_DIR=""
fi
CLAUDE_DIR="${CLAUDE_HOME:-$HOME/.claude}"
HOOKS_DIR="$CLAUDE_DIR/hooks/attention-span"
RELEASES_DIR="$HOOKS_DIR/releases"
SETTINGS="$CLAUDE_DIR/settings.json"
SETTINGS_ORIGINAL="$SETTINGS.statusline-original"
SETTINGS_CREATED="$SETTINGS.statusline-created-by-attention-span"

c_green()  { printf '\033[0;32m%s\033[0m\n' "$1"; }
c_yellow() { printf '\033[0;33m%s\033[0m\n' "$1"; }
c_red()    { printf '\033[0;31m%s\033[0m\n' "$1"; }
c_dim()    { printf '\033[2m%s\033[0m\n'    "$1"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { c_red "Missing required command: $1"; exit 1; }
}

uninstall() {
  c_yellow "Uninstalling attention-span statusline..."

  if [[ -f "$SETTINGS" ]]; then
    settings_message=$(python3 - "$SETTINGS" "$SETTINGS_ORIGINAL" "$SETTINGS_CREATED" \
      "$HOOKS_DIR" <<'PYEOF'
import glob
import json
import os
import sys
import tempfile

p, original_path, created_path, hooks_dir = sys.argv[1:]

def load(path):
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("settings root must be a JSON object")
    return value

def owned(value):
    if not isinstance(value, dict):
        return False
    command = value.get("command", "")
    targets = tuple(os.path.join(hooks_dir, name) for name in
                    ("launcher.py", "statusline.py"))
    return isinstance(command, str) and any(target in command for target in targets)

def write(path, value):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".cc-health-settings-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

try:
    current = load(p)
except Exception as exc:
    print("Cannot safely update {}: {}".format(p, exc), file=sys.stderr)
    sys.exit(2)

# A user may have replaced the statusline after installation. Never overwrite that choice.
message = "Kept the current statusLine setting."
if owned(current.get("statusLine")):
    original = None
    if os.path.isfile(original_path):
        try:
            original = load(original_path)
        except Exception as exc:
            print("Cannot read saved settings {}: {}".format(original_path, exc), file=sys.stderr)
            sys.exit(2)
    else:
        legacy = sorted(glob.glob(p + ".statusline-backup-*"))
        if legacy:
            try:
                original = load(legacy[0])
            except Exception:
                original = None

    if original is not None and "statusLine" in original:
        current["statusLine"] = original["statusLine"]
        message = "Restored the previous statusLine setting."
    else:
        current.pop("statusLine", None)
        message = "Removed the attention-span setting."
    write(p, current)

for path in (original_path, created_path):
    try:
        os.unlink(path)
    except OSError:
        pass
print(message)
PYEOF
)
    c_green "$settings_message"
  else
    rm -f "$SETTINGS_ORIGINAL" "$SETTINGS_CREATED"
  fi

  rm -rf "$HOOKS_DIR"
  c_green "Removed installed scripts."
  c_dim "Your Claude Code transcripts (~/.claude/projects/) were not touched."
  exit 0
}

if [[ "${1:-}" == "--uninstall" ]]; then
  uninstall
fi

if [[ -z "$REPO_DIR" || ! -f "$REPO_DIR/attention_span/statusline.py" ]]; then
  if [[ "${ATTENTION_SPAN_BOOTSTRAP:-0}" == "1" ]]; then
    c_red "The downloaded attention-span archive is incomplete; aborting." >&2
    exit 1
  fi
  require curl
  require tar
  TARBALL_URL="${ATTENTION_SPAN_TARBALL_URL:-https://github.com/lobel-dev/attention-span/archive/refs/heads/main.tar.gz}"
  BOOTSTRAP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/attention-span-install.XXXXXX")
  trap 'rm -rf "$BOOTSTRAP_DIR"' EXIT
  c_dim "Downloading attention-span..."
  curl -fsSL "$TARBALL_URL" | tar -xz -C "$BOOTSTRAP_DIR" --strip-components=1
  ATTENTION_SPAN_BOOTSTRAP=1 bash "$BOOTSTRAP_DIR/install.sh" "$@"
  exit $?
fi

require python3
require bash

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  c_red "Python 3.11 or newer is required; found $(python3 --version 2>&1 || printf 'an unsupported interpreter')." >&2
  exit 1
fi

VERSION_VALUE=$(
  cd -- "$REPO_DIR" && PYTHONPATH='' PYTHONSAFEPATH='' python3 - "$REPO_DIR" <<'PYEOF'
import sys

from attention_span.release_contract import parse_version

root = sys.argv[1]
with open(root + "/VERSION") as f:
    version = f.read().strip()
try:
    parse_version(version)
except ValueError:
    print("VERSION must be strict ASCII SemVer X.Y.Z", file=sys.stderr)
    sys.exit(2)
print(version)
PYEOF
)

if [[ ! -d "$CLAUDE_DIR" ]]; then
  c_red "$CLAUDE_DIR does not exist."
  c_dim "Is Claude Code installed? Run it once before installing this statusline."
  exit 1
fi

if [[ -f "$SETTINGS" ]]; then
  python3 - "$SETTINGS" <<'PYEOF'
import json
import sys

path = sys.argv[1]
try:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("settings root must be a JSON object")
except Exception as exc:
    print("Cannot safely update {}: {}".format(path, exc), file=sys.stderr)
    sys.exit(2)
PYEOF
fi

mkdir -p "$HOOKS_DIR"
mkdir -p "$RELEASES_DIR"

RELEASE_ID="$VERSION_VALUE-install-$(date +%Y%m%d%H%M%S)-$$"
STAGE_DIR=$(mktemp -d "$RELEASES_DIR/.$RELEASE_ID.XXXXXX")
mkdir -p "$STAGE_DIR/attention_span"

PAYLOAD_FILES=(
  LICENSE
  VERSION
  attention_span/__init__.py
  attention_span/agent_health.py
  attention_span/analysis.py
  attention_span/detectors.py
  attention_span/events.py
  attention_span/health_config.py
  attention_span/reducer.py
  attention_span/release_contract.py
  attention_span/render.py
  attention_span/render_facts.py
  attention_span/session_ui.py
  attention_span/status_catalog.py
  attention_span/statusline.py
  attention_span/subagents.py
  attention_span/text.py
  attention_span/transcript.py
  attention_span/update.py
  attention_span/verdicts.py
  launcher.py
  statusline.py
  update.py
)
EXECUTABLE_FILES=(launcher.py statusline.py update.py)

for relative in "${PAYLOAD_FILES[@]}"; do
  cp "$REPO_DIR/$relative" "$STAGE_DIR/$relative"
done
for relative in "${EXECUTABLE_FILES[@]}"; do
  chmod +x "$STAGE_DIR/$relative"
done
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
mv "$STAGE_DIR" "$RELEASE_DIR"

python3 - "$HOOKS_DIR" "$RELEASE_ID" <<'PYEOF'
import os
import secrets
import sys

root, release_id = sys.argv[1:]

def replace_symlink(name, target):
    temporary = os.path.join(
        root, ".{}-{}-{}".format(name, os.getpid(), secrets.token_hex(4))
    )
    try:
        os.symlink(target, temporary)
        os.replace(temporary, os.path.join(root, name))
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass

has_current = os.path.islink(os.path.join(root, "current"))
aliases = (
    ("launcher.py", os.path.join("current", "launcher.py")),
    ("VERSION", os.path.join("current", "VERSION")),
)

# On reinstall, publish aliases first: they continue to resolve through the old current
# until its one atomic switch. On a flat/fresh install, publish current first so no alias
# is ever left dangling while an old settings entry may still be active.
if has_current:
    for name, target in aliases:
        replace_symlink(name, target)
replace_symlink("current", os.path.join("releases", release_id))
if not has_current:
    for name, target in aliases:
        replace_symlink(name, target)
PYEOF
c_green "✓ Installed release $VERSION_VALUE → $HOOKS_DIR/current"
c_dim "  Runtime layout: launcher.py → current/statusline.py → attention_span package."

TARGET_CMD="python3 \"$HOOKS_DIR/launcher.py\""

python3 - "$SETTINGS" "$TARGET_CMD" "$SETTINGS_ORIGINAL" "$SETTINGS_CREATED" \
  "$HOOKS_DIR" <<'PYEOF'
import glob
import json
import os
import shutil
import sys
import tempfile

p, command, original_path, created_path, hooks_dir = sys.argv[1:]

def load(path):
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("settings root must be a JSON object")
    return value

def owned(value):
    if not isinstance(value, dict):
        return False
    old_command = value.get("command", "")
    targets = tuple(os.path.join(hooks_dir, name) for name in
                    ("launcher.py", "statusline.py"))
    return isinstance(old_command, str) and any(target in old_command for target in targets)

exists = os.path.isfile(p)
if exists:
    try:
        settings = load(p)
    except Exception as exc:
        print("Cannot safely update {}: {}".format(p, exc), file=sys.stderr)
        sys.exit(2)
else:
    settings = {}

# Save the pre-install state once. Reinstalling never replaces this recovery point.
if not os.path.exists(original_path) and not os.path.exists(created_path):
    if exists and owned(settings.get("statusLine")):
        legacy = sorted(glob.glob(p + ".statusline-backup-*"))
        if legacy:
            shutil.copyfile(legacy[0], original_path)
        else:
            open(created_path, "a").close()
    elif exists:
        shutil.copyfile(p, original_path)
    else:
        open(created_path, "a").close()

settings["statusLine"] = {"type": "command", "command": command}
directory = os.path.dirname(p) or "."
fd, tmp = tempfile.mkstemp(prefix=".cc-health-settings-", dir=directory, text=True)
try:
    with os.fdopen(fd, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)
except Exception:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
PYEOF
c_dim "  Saved the original state for uninstall."
c_green "✓ Updated Claude Code settings"

echo
c_green "Attention Span status line is installed."
c_dim "Automatic stable-release updates: on (daily detached check)."
c_dim "Opt out: export CLAUDE_HEALTH_AUTO_UPDATE=0"
c_dim "Open a new Claude Code session."
c_dim "Help: https://github.com/lobel-dev/attention-span#readme"
c_dim "Uninstall: curl -fsSL https://raw.githubusercontent.com/lobel-dev/attention-span/main/install.sh | bash -s -- --uninstall"
echo
