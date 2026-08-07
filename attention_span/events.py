"""Classify transcript tool and user events."""

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

BashClassification: TypeAlias = Literal["read", "edit", "neutral"]
ErrorClassification: TypeAlias = Literal["OK", "HOOK_DENY", "GENUINE"]

READ_TOOLS = {"Read", "Grep", "Glob"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

BASH_EDIT_HEADS = {
    "rm",
    "rmdir",
    "mv",
    "cp",
    "mkdir",
    "touch",
    "chmod",
    "chown",
    "ln",
    "dd",
    "truncate",
    "patch",
    "install",
    "tee",
}

BASH_READ_HEADS = {
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "bat",
    "ls",
    "find",
    "fd",
    "grep",
    "rg",
    "ag",
    "wc",
    "stat",
    "file",
    "tree",
    "diff",
    "cmp",
    "du",
    "df",
    "awk",
    "pwd",
    "which",
    "type",
    "realpath",
    "readlink",
}
BASH_DUMP_HEADS = {"cat", "head", "tail", "less", "more", "bat", "sed"}

BASH_CHDIR_HEADS = {"cd", "pushd", "popd"}

GIT_EDIT_SUBS = {
    "add",
    "commit",
    "rm",
    "mv",
    "checkout",
    "restore",
    "reset",
    "stash",
    "apply",
    "push",
    "merge",
    "rebase",
    "clean",
    "cherry-pick",
    "revert",
}
GIT_READ_SUBS = {"status", "diff", "log", "show", "blame", "ls-files", "branch"}
PKG_EDIT_SUBS = {
    "install",
    "add",
    "i",
    "ci",
    "remove",
    "uninstall",
    "update",
    "upgrade",
}
PIP_EDIT_SUBS = {"install", "uninstall"}

_CMD_PREFIXES = {
    "sudo",
    "env",
    "time",
    "nohup",
    "command",
    "builtin",
    "exec",
    "nice",
    "ionice",
    "stdbuf",
    "setsid",
    "doas",
}
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_REDIR_RE = re.compile(r"(\d*&?>>?)\s*(\S+)")
_HEREDOC_RE = re.compile(r"<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?")
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||[;|\n]")
_SHELL_META_RE = re.compile(r"""[*?\[\]{}()<>$`'"\\!~=,\s]""")
_PATH_TOKEN_RE = re.compile(r"[^/.]\.[A-Za-z0-9_]{1,10}$")
_SYNTHETIC_PREFIXES = (
    "<command-message>",
    "<command-name>",
    "<command-args>",
    "<task-notification>",
    "<local-command-caveat>",
    "<system-reminder>",
    "<user-prompt-submit-hook>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",  # defensive: travels with <bash-stdout>, not seen leading alone
    "<local-command-stdout>",
    "Unknown command:",  # system echo for failed slash commands
    "Caveat:",  # local-command caveat preamble
)


def _is_synthetic(text: str | None) -> bool:
    """True if a user-message text is harness-generated, not a real human turn."""
    if not text:
        return True
    return text.lstrip().startswith(_SYNTHETIC_PREFIXES)


def _strip_heredocs(command: str) -> str:
    """Remove here-doc bodies (they are data, not commands), keeping the opening
    line so its redirects (e.g. `cat <<EOF > file`) still count."""
    if "<<" not in command:  # no heredoc opener -> nothing to strip (the common case)
        return command
    out = []
    delim = None
    for line in command.split("\n"):
        if delim is not None:
            if line.strip() == delim:
                delim = None
            continue  # drop body + terminator
        out.append(line)
        m = _HEREDOC_RE.search(line)
        if m:
            delim = m.group(1)
    return "\n".join(out)


def _has_inplace_flag(rest: list[str]) -> bool:
    """True if a sed/perl arg list requests in-place editing (-i / --in-place)."""
    for a in rest:
        if a == "--in-place" or a.startswith("--in-place="):
            return True
        if a.startswith("-") and not a.startswith("--") and "i" in a:
            return True
    return False


def _segment_head(seg: str) -> tuple[str, list[str]]:
    """Return (head_command_basename, tokens_from_head_onward) for one segment,
    skipping leading env assignments and wrapper prefixes (sudo, env, time...)."""
    toks = seg.split()
    i = 0
    while i < len(toks) and (_ENV_ASSIGN_RE.match(toks[i]) or toks[i] in _CMD_PREFIXES):
        i += 1
    if i >= len(toks):
        return "", []
    return toks[i].split("/")[-1], toks[i:]


def _classify_segment(seg: str) -> BashClassification:
    """Classify a single shell segment as read | edit | neutral."""
    seg = seg.strip()
    if not seg:
        return "neutral"

    for m in _REDIR_RE.finditer(seg):
        target = m.group(2)
        if not target or target in ('"', "'", "`"):
            continue
        if target == "/dev/null" or target.startswith("&"):
            continue
        return "edit"

    head, toks = _segment_head(seg)
    if not head:
        return "neutral"
    rest = toks[1:]
    sub = rest[0] if rest else ""  # first arg; used by the subcommand-keyed heads below

    if head == "sed":
        return "edit" if _has_inplace_flag(rest) else "read"
    if head == "perl":
        return "edit" if _has_inplace_flag(rest) else "neutral"
    if head == "git":
        if sub in GIT_EDIT_SUBS:
            return "edit"
        if sub in GIT_READ_SUBS:
            return "read"
        return "neutral"
    if head in ("npm", "yarn", "pnpm"):
        return "edit" if sub in PKG_EDIT_SUBS else "neutral"
    if head in ("pip", "pip3"):
        return "edit" if sub in PIP_EDIT_SUBS else "neutral"

    if head in BASH_EDIT_HEADS:
        return "edit"
    if head in BASH_READ_HEADS:
        return "read"
    return "neutral"  # builds, tests, runners, cd, echo — fail-safe bucket


def classify_bash(command: Any) -> BashClassification:
    """Classify a Bash command as read | edit | neutral.

    Mutation dominates and is sticky: if any segment writes (an edit head or a
    real `>`/`>>` redirect to a path), the whole command is an edit. Otherwise a
    read if any segment reads, else neutral. Here-doc bodies are stripped first.
    """
    if not command or not isinstance(command, str):
        return "neutral"
    text = _strip_heredocs(command)
    saw_read = False
    for seg in _SEGMENT_SPLIT_RE.split(text):
        c = _classify_segment(seg)
        if c == "edit":
            return "edit"
        if c == "read":
            saw_read = True
    return "read" if saw_read else "neutral"


def classify_tool(
    name: Any, tool_input: Mapping[str, Any] | None
) -> BashClassification:
    """Classify a tool_use as read | edit | neutral.

    Task/Agent/TodoWrite/ToolSearch/AskUserQuestion/Task*/mcp__*/unknown all fall
    through to neutral — they are not read/edit behavior and never advance the
    window.
    """
    if name in READ_TOOLS:
        return "read"
    if name in EDIT_TOOLS:
        return "edit"
    if name == "Bash":
        return classify_bash((tool_input or {}).get("command", ""))
    return "neutral"


def _is_path_token(tok: str) -> bool:
    """True if a shell token is a literal, extension-bearing file path."""
    if not tok or tok.startswith("-"):  # option flag, never an operand
        return False
    if _SHELL_META_RE.search(tok):
        return False
    return bool(_PATH_TOKEN_RE.search(tok))


def _read_anchor(segments: Sequence[str], cwd: Any) -> str | None:
    """The directory a RELATIVE shell-read operand may resolve against, else None.

    None means "refuse to anchor", leaving relative operands unresolvable as before.
    Two conditions must both hold to anchor: the record carries a non-empty ABSOLUTE
    `cwd` string (an off-schema or relative one tells us nothing), and NO segment of
    the command changes directory (BASH_CHDIR_HEADS) - see that set for why. Leading
    shell grouping chars are stripped before the head is matched so `(cd sub && cat f)`
    guards too.
    """
    if not isinstance(cwd, str) or not os.path.isabs(cwd):
        return None
    for seg in segments:
        if _segment_head(seg)[0].lstrip("({") in BASH_CHDIR_HEADS:
            return None
    return cwd


def _bash_read_path(command: str, cwd: Any = None) -> str | None:
    """The one file a read-classified shell command dumps, normalized; else None.

    Exists so an agent that rereads a failed file through the shell (`cat a.py`)
    clears the blind-loop detector's pending failure exactly as a Read tool call
    does. Wrong beats missing here ONLY in one direction: a path that accidentally
    matched a pending file would silently suppress a TRUE alarm, so every ambiguity
    resolves to None (status quo) instead of to a guess. Four rules enforce that:

      * only BASH_DUMP_HEADS segments are inspected (heads whose operands are files
        whose content is dumped), split the same way ``classify_bash`` splits, so a
        pipeline reads consistently under both;
      * only literal extension-bearing tokens count as operands (``_is_path_token``),
        which drops flags, flag values, sed scripts, globs and directories;
      * a RELATIVE operand is anchored to the record's ``cwd`` - the common spelling
        of the reread is `cat config.json`, which must still equal the Edit tool's
        absolute `/repo/config.json`. When ``_read_anchor`` refuses (no usable cwd, or
        the command moves directory) the whole command yields None rather than a
        half-resolved guess: an unresolvable operand also destroys the one-file fact
        the next rule needs;
      * the whole command must name exactly ONE distinct file, counted AFTER anchoring.
        Zero (`cat` of a pathless stream) and many (`head a.py b.py`) both yield None -
        an event carries a single file_path, and picking one of several would be a guess.
    """
    segments = [
        seg.strip() for seg in _SEGMENT_SPLIT_RE.split(_strip_heredocs(command))
    ]
    anchor = _read_anchor(segments, cwd)
    found: set[str] = set()
    for seg in segments:
        head, toks = _segment_head(seg)
        if head not in BASH_DUMP_HEADS:
            continue
        for tok in toks[1:]:
            if not _is_path_token(tok):
                continue
            if not os.path.isabs(tok):
                if anchor is None:
                    return None
                tok = os.path.join(anchor, tok)
            found.add(os.path.normpath(tok))
            if len(found) > 1:
                return None
    return found.pop() if len(found) == 1 else None


def _tool_file_path(
    name: Any, tool_input: Mapping[str, Any] | None, cwd: Any = None
) -> str | None:
    """The file a read/edit tool touches, normalized; None if not file-scoped.

    ``cwd`` is the record's own working directory (the transcript line's `cwd` field);
    it anchors relative operands of a shell read and is ignored by every other tool -
    Edit/Read file_paths already arrive absolute. Off-schema or missing is fine: the
    extractor then simply declines to anchor.
    """
    inp = tool_input or {}
    p = None
    if name == "NotebookEdit":
        p = inp.get("notebook_path") or inp.get("file_path")
    elif name in EDIT_TOOLS or name == "Read":
        p = inp.get("file_path")
    elif name == "Bash":
        # File-scoped shell READS only. An edit-classified command keeps no path: a
        # mutation's target is not a reread, and handing one to the blind-loop
        # detector would let `touch a.py` masquerade as a failed edit of a.py.
        cmd = inp.get("command")
        if isinstance(cmd, str) and classify_bash(cmd) == "read":
            return _bash_read_path(cmd, cwd)
    if not p or not isinstance(p, str):
        return None
    return os.path.normpath(p)


def _normalize_content(content: Any) -> str:
    """Flatten a tool_result content field (str or list of text parts) to a str.

    A part's `text` is documented as a str but JSON permits an off-schema value
    (null or a number). Coerce rather than pass through: a non-str item would
    raise inside the join, and analyze_transcript's broad guard swallows it —
    silently freezing every metric at the bad line. `str(... or "")` maps
    None/falsy to "" and stringifies the rest.
    """
    if isinstance(content, list):
        return "".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
    return str(content or "")


def _looks_like_hook_deny(body: str) -> bool:
    """True if an error body is a hook/gate denial rather than a genuine failure."""
    s = body.lstrip()
    low = body.lower()
    if s.startswith("[") and "Gate" in body:
        return True
    if "GateGuard" in body or "PreToolUse" in body:
        return True
    if "blocked by" in low:
        return True
    return bool("hook" in low and "block" in low)


def classify_error(is_error: Any, content: Any) -> ErrorClassification:
    """Classify a tool_result's provenance: OK | HOOK_DENY | GENUINE.

    `<tool_use_error>` is Claude Code's authoritative marker for a genuine tool
    failure and wins outright. Otherwise hook/gate denials are filtered out so
    they neither count as failed edits nor advance the R2E window. Any remaining
    `is_error` (e.g. a bash non-zero exit) defaults to GENUINE.
    """
    if not is_error:
        return "OK"
    body = _normalize_content(content)
    if "<tool_use_error>" in body:
        return "GENUINE"
    if _looks_like_hook_deny(body):
        return "HOOK_DENY"
    return "GENUINE"
