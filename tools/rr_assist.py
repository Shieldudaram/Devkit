#!/usr/bin/env python3
"""
rr_assist.py - repo-aware assistant for Hytale mods (plan -> patch), budgeted for cost control.

Usage (inside container):
  cd /workspace/<mod-repo>
  python /workspace/devkit/tools/rr_assist.py plan -m feature "Implement X" --auto --grep "Symbol"
  python /workspace/devkit/tools/rr_assist.py patch
  git apply --check .rr_assist/patch.diff && git apply .rr_assist/patch.diff && ./gradlew build

Notes:
- Runs against the git repo you are currently in (works great with /workspace as a parent folder of many repos).
- Refuses to generate patches that touch docker/gradle config by default.
- Uses rg if installed; falls back to grep.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from openai import OpenAI

# ---------- Environment Helpers ----------
# We avoid external dependencies like python-dotenv by implementing a simple
# .env loader. This will read key=value pairs from a `.env` file at the repo
# root and populate os.environ accordingly (if those keys are not already set).
def load_dotenv_file(repo_root: Path) -> None:
    """
    Load environment variables from a .env file in the repository root.
    Lines starting with '#' are ignored. Key-value pairs are split on the first '='.
    Only keys not already present in os.environ are set.
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # Fail silently if .env cannot be read
        pass

# Approximate token estimator. OpenAI tokenizers vary, but a rough rule of thumb
# is ~4 characters per token in English. We use this to enforce simple budgets.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------- Memory / Session Helpers ----------

def get_global_state_dir() -> Path:
    """A shared state directory across all repos (best-effort)."""
    env = os.environ.get("RR_ASSIST_GLOBAL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # Prefer a shared /workspace volume if present.
    ws = Path("/workspace")
    if ws.exists():
        return (ws / ".rr_assist_global").resolve()
    return (Path.home() / ".rr_assist_global").resolve()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_text(p: Path, max_chars: int = 50_000) -> str:
    if not p.exists():
        return ""
    try:
        txt = p.read_text(encoding="utf-8")
        if len(txt) > max_chars:
            return txt[-max_chars:]
        return txt
    except Exception:
        return ""


def append_and_cap(p: Path, content: str, max_chars: int = 60_000) -> None:
    try:
        existing = safe_read_text(p, max_chars=max_chars)
        combined = (existing + "\n\n" + content).strip() + "\n"
        if len(combined) > max_chars:
            combined = combined[-max_chars:]
        p.write_text(combined, encoding="utf-8")
    except Exception:
        pass


def memory_blob(repo_root: Path, memory_scope: str, session_scope: str) -> str:
    """Build a memory blob from repo + global memory/session files."""
    parts: List[str] = []
    repo_state = ensure_state_dir(repo_root)
    global_state = ensure_dir(get_global_state_dir())

    if memory_scope in ("repo", "both"):
        repo_mem = safe_read_text(repo_state / "memory.md")
        if repo_mem.strip():
            parts.append("## REPO MEMORY\n" + repo_mem.strip())

    if memory_scope in ("global", "both"):
        global_mem = safe_read_text(global_state / "memory.md")
        if global_mem.strip():
            parts.append("## GLOBAL MEMORY\n" + global_mem.strip())

    if session_scope in ("repo", "both"):
        repo_sess = safe_read_text(repo_state / "current_context.md")
        if repo_sess.strip():
            parts.append("## REPO SESSION\n" + repo_sess.strip())

    if session_scope in ("global", "both"):
        global_sess = safe_read_text(global_state / "current_context.md")
        if global_sess.strip():
            parts.append("## GLOBAL SESSION\n" + global_sess.strip())

    return "\n\n".join(parts).strip()


# ---------- Defaults / Policy ----------

# Files we do NOT want the model to modify (patch validator will refuse).
DEFAULT_FORBIDDEN_PATTERNS = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".dockerignore",
    "**/Dockerfile",
    "**/docker-compose.yml",
    "**/docker-compose.yaml",
    "**/.dockerignore",

    "build.gradle",
    "settings.gradle",
    "gradle.properties",
    "gradlew",
    "gradlew.bat",
    "gradle/**",
    "**/build.gradle",
    "**/settings.gradle",
    "**/gradle.properties",
    "**/gradlew",
    "**/gradlew.bat",
    "**/gradle/**",
]

# Exclusions to avoid sending huge/binary/generated stuff.
DEFAULT_EXCLUDES = [
    ".git/**",
    ".gradle/**",
    "build/**",
    "**/build/**",
    "out/**",
    "**/*.jar",
    "**/*.zip",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.webp",
    "**/*.gif",
    "**/*.mp3",
    "**/*.ogg",
    "**/*.wav",
    "**/*.bin",
    "**/*.aot",
    "**/*.class",
    "**/*.so",
    "**/*.dylib",
    "**/*.exe",
]

PROJECT_CONTEXT_PATHS = [
    Path("docs/PROJECT_CONTEXT.md"),
    Path("PROJECT_CONTEXT.md"),
]

STATE_DIRNAME = ".rr_assist"
LAST_REQ_NAME = "last_request.json"
LAST_PLAN_NAME = "last_plan.md"
HISTORY_NAME = "history.jsonl"
DEFAULT_PATCH_OUT = ".rr_assist/patch.diff"


# ---------- Helpers ----------

def sh(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr.strip()}")
    return (p.stdout or "").strip()

def in_git_repo() -> bool:
    try:
        return sh(["git", "rev-parse", "--is-inside-work-tree"], check=True) == "true"
    except Exception:
        return False

def git_root() -> Path:
    return Path(sh(["git", "rev-parse", "--show-toplevel"], check=True))

def ensure_state_dir(repo_root: Path) -> Path:
    d = repo_root / STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_history(repo_root: Path, entry: dict) -> None:
    d = ensure_state_dir(repo_root)
    p = d / HISTORY_NAME
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_last_request(repo_root: Path, payload: dict) -> None:
    d = ensure_state_dir(repo_root)
    (d / LAST_REQ_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_last_request(repo_root: Path) -> dict:
    d = ensure_state_dir(repo_root)
    return json.loads((d / LAST_REQ_NAME).read_text(encoding="utf-8"))

def save_last_plan(repo_root: Path, plan_text: str) -> None:
    d = ensure_state_dir(repo_root)
    (d / LAST_PLAN_NAME).write_text(plan_text.strip() + "\n", encoding="utf-8")

def load_last_plan(repo_root: Path) -> str:
    d = ensure_state_dir(repo_root)
    return (d / LAST_PLAN_NAME).read_text(encoding="utf-8")

def project_context(repo_root: Path, max_chars: int) -> str:
    for rel in PROJECT_CONTEXT_PATHS:
        p = repo_root / rel
        if p.exists():
            return read_text(p, max_chars=max_chars)
    return ""

def path_matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    p = Path(rel_path)
    return any(p.match(pat) for pat in patterns)

def read_text(path: Path, max_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_chars:
            return data[:max_chars] + "\n\n[TRUNCATED]\n"
        return data
    except Exception as e:
        return f"[ERROR reading {path}: {e}]"

def is_rg_available() -> bool:
    return shutil.which("rg") is not None

def grep_search(repo_root: Path, query: str) -> str:
    """
    Uses rg if available, else grep. Returns raw output (may be large; caller should truncate).
    """
    if is_rg_available():
        try:
            return sh(["rg", "-n", "--hidden", "--glob", "!.git/*", query, "."], cwd=repo_root, check=True)
        except Exception as e:
            return f"[rg failed: {e}]"
    else:
        # grep fallback (slower but works)
        try:
            return sh([
                "grep", "-RIn",
                "--exclude-dir=.git",
                "--exclude-dir=build",
                "--exclude-dir=.gradle",
                query, "."
            ], cwd=repo_root, check=True)
        except Exception as e:
            return f"[grep failed: {e}]"

def git_status(repo_root: Path) -> str:
    return sh(["git", "status", "--porcelain", "-b"], cwd=repo_root, check=True)

def git_diff_stat(repo_root: Path) -> str:
    return sh(["git", "diff", "--stat"], cwd=repo_root, check=True)

def git_diff(repo_root: Path) -> str:
    return sh(["git", "diff"], cwd=repo_root, check=True)

def changed_files(repo_root: Path) -> List[str]:
    out = sh(["git", "status", "--porcelain"], cwd=repo_root, check=True)
    files = []
    for line in out.splitlines():
        if len(line) >= 4:
            rel = line[3:].strip()
            if rel:
                files.append(rel)
    # unique preserve order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq

def discover_entrypoints(repo_root: Path, max_hits: int = 8) -> List[str]:
    """
    Lightweight discovery (no file content sent yet): look for plugin entry points.
    Works even without rg (uses grep).
    """
    patterns = [
        r"extends\s+JavaPlugin",
        r"JavaPluginInit",
        r"\bonEnable\b",
    ]
    hits: List[str] = []
    for pat in patterns:
        out = grep_search(repo_root, pat)
        # Extract file paths from grep/rg style output: path:line:content
        for line in out.splitlines():
            if ":" in line:
                rel = line.split(":", 1)[0]
                # Basic filter: only likely source files
                if rel.endswith(".java") or rel.endswith(".kt"):
                    hits.append(rel)
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break

    # unique preserve order + exists
    seen = set()
    uniq = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        p = repo_root / h
        if p.exists() and p.is_file():
            uniq.append(h)
    return uniq[:max_hits]

@dataclass
class Budget:
    max_files: int
    max_file_chars: int
    max_total_chars: int
    max_grep_chars: int

def collect_files(
    repo_root: Path,
    globs: Optional[List[str]],
    include_changed: bool,
    auto: bool,
    excludes: List[str],
    budget: Budget
) -> Tuple[List[str], List[str]]:
    """
    Returns (selected_files, notes_about_omissions)
    """
    candidates: List[str] = []

    if include_changed:
        candidates.extend(changed_files(repo_root))

    if globs:
        for g in globs:
            for p in repo_root.glob(g):
                if p.is_file():
                    candidates.append(str(p.relative_to(repo_root)))

    if auto and not globs and not include_changed:
        # Auto mode: include discovered entrypoints (and no more than a few).
        candidates.extend(discover_entrypoints(repo_root, max_hits=8))

    # unique preserve order
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if path_matches_any(c, excludes):
            continue
        if (repo_root / c).is_file():
            uniq.append(c)

    notes: List[str] = []
    if len(uniq) > budget.max_files:
        notes.append(f"[Omitted {len(uniq) - budget.max_files} files due to --max-files={budget.max_files}]")
        uniq = uniq[: budget.max_files]

    return uniq, notes

def build_context(
    repo_root: Path,
    mode: str,
    request_text: str,
    plan_text: Optional[str],
    files: List[str],
    grep_queries: List[str],
    include_full_diff: bool,
    budget: Budget
) -> Tuple[str, List[str]]:
    """
    Returns (context_blob, notes). Notes include truncation/omission info.
    """
    notes: List[str] = []

    pc = project_context(repo_root, max_chars=min(30_000, budget.max_file_chars))
    st = git_status(repo_root)
    ds = git_diff_stat(repo_root)

    parts: List[str] = []
    if pc.strip():
        parts.append("## PROJECT_CONTEXT\n" + pc.strip())
    parts.append("## GIT_STATUS\n" + st.strip())
    parts.append("## GIT_DIFF_STAT\n" + (ds.strip() if ds.strip() else "[no diff]"))

    if include_full_diff:
        d = git_diff(repo_root)
        # Full diff can be big; trim to remaining budget later.
        parts.append("## GIT_DIFF\n" + (d if d else "[no diff]"))

    if grep_queries:
        blob = []
        for q in grep_queries:
            out = grep_search(repo_root, q)
            if len(out) > budget.max_grep_chars:
                out = out[:budget.max_grep_chars] + "\n[TRUNCATED GREP OUTPUT]\n"
                notes.append(f"[Truncated grep output for query={q!r} to --max-grep-chars={budget.max_grep_chars}]")
            blob.append(f"### search: {q}\n{out}")
        parts.append("## SEARCH\n" + "\n\n".join(blob))

    if files:
        file_blobs: List[str] = []
        for rel in files:
            content = read_text(repo_root / rel, max_chars=budget.max_file_chars)
            file_blobs.append(f"### FILE: {rel}\n{content}")
            if content.endswith("[TRUNCATED]\n"):
                notes.append(f"[Truncated file {rel} to --max-file-chars={budget.max_file_chars}]")
        parts.append("## FILES\n" + "\n\n".join(file_blobs))

    if plan_text is not None:
        parts.append("## PLAN\n" + plan_text.strip())

    # Now enforce max_total_chars across everything
    combined = "\n\n".join(parts).strip()
    if len(combined) > budget.max_total_chars:
        combined = combined[:budget.max_total_chars] + "\n\n[TRUNCATED CONTEXT: max_total_chars reached]\n"
        notes.append(f"[Truncated total context to --max-total-chars={budget.max_total_chars}]")

    return combined, notes

def extract_unified_diff(text: str) -> Optional[str]:
    # Start from first diff header
    m = re.search(r"^diff --git\s", text, flags=re.MULTILINE)
    if not m:
        return None
    return text[m.start():].strip() + "\n"

def touched_paths_from_diff(diff_text: str) -> List[str]:
    touched = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a = parts[2].removeprefix("a/")
                b = parts[3].removeprefix("b/")
                if a != "/dev/null":
                    touched.append(a)
                if b != "/dev/null":
                    touched.append(b)
    # unique preserve order
    seen = set()
    out = []
    for p in touched:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def call_openai(
    model: str,
    effort: str,
    system: str,
    user: str,
    *,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Pass it via docker-compose env/env_file.")

    client = OpenAI()
    kwargs = {}
    if effort and effort != "none":
        kwargs["reasoning"] = {"effort": effort}

    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    txt = getattr(resp, "output_text", None)
    rid = getattr(resp, "id", None)
    return (txt or "").strip(), rid


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Repo-aware assistant (plan -> patch), budgeted.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-m", "--mode", default="feature", choices=["feature", "debug", "refactor", "docs"])
    common.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.1"))
    common.add_argument("--effort", default=os.environ.get("OPENAI_EFFORT", "none"),
                        choices=["none", "low", "medium", "high"])

    common.add_argument("--files", nargs="*", help="Glob(s) of files to include (relative to repo root)")
    common.add_argument("--include-changed", action="store_true", help="Include files changed per git status")
    common.add_argument("--auto", action="store_true",
                        help="Auto-include likely entrypoint files if no --files/--include-changed are provided")

    common.add_argument("--diff", action="store_true", help="Include full git diff (can be big)")
    common.add_argument("--grep", action="append", default=[], help="Search query (uses rg if available, else grep)")

    common.add_argument("--max-files", type=int, default=30)
    common.add_argument("--max-file-chars", type=int, default=20_000)
    common.add_argument("--max-total-chars", type=int, default=120_000)
    common.add_argument("--max-grep-chars", type=int, default=20_000)

    # Run from a parent folder by targeting a repo explicitly.
    common.add_argument("--repo", default=None,
                        help="Path to target git repo (if you are running from a parent folder).")

    # Memory/session settings.
    common.add_argument("--memory-scope", default="both", choices=["none", "repo", "global", "both"],
                        help="Which persistent memory files to include.")
    common.add_argument("--session-scope", default="repo", choices=["none", "repo", "global", "both"],
                        help="Which rolling session context files to include.")
    common.add_argument("--thread-scope", default="repo", choices=["none", "repo", "global"],
                        help="Which conversation thread to continue via previous_response_id.")
    common.add_argument("--reset-thread", action="store_true", help="Start a new thread (forget previous_response_id).")
    common.add_argument("--reset-session", action="store_true", help="Clear rolling session context before running.")

    # Additional flags for extended functionality.
    common.add_argument("--max-tokens", type=int, default=None,
                        help="Approximate maximum tokens allowed for a request (input).")
    common.add_argument("--include-default", action="store_true",
                        help="Include common important files like README.md, settings.gradle and manifest.json by default.")
    common.add_argument("--attempts", type=int, default=1,
                        help="Number of attempts to refine the patch by running build/tests (patch mode only).")

    common.add_argument("--no-default-excludes", action="store_true")
    common.add_argument("--allow-forbidden", action="store_true",
                        help="Allow patch touching docker/gradle config files (not recommended)")

    p_plan = sub.add_parser("plan", parents=[common], help="Generate a plan (no code/diff).")
    p_plan.add_argument("prompt", help="What you want to do")
    p_plan.add_argument("--save-context", action="store_true", help="Save the final context blob to .rr_assist/context.txt")

    p_patch = sub.add_parser("patch", parents=[common], help="Generate a unified diff patch from last plan.")
    p_patch.add_argument("-o", "--out", default=DEFAULT_PATCH_OUT, help="Patch output path (relative to repo root)")
    p_patch.add_argument("--save-context", action="store_true", help="Save the final context blob to .rr_assist/context.txt")

    p_note = sub.add_parser("note", help="Append a note to memory (repo or global).")
    p_note.add_argument("--scope", default="repo", choices=["repo", "global"],
                        help="Which memory file to append to.")
    p_note.add_argument("text", help="The note to append")

    p_reset = sub.add_parser("reset", help="Reset session/thread state for repo and/or global.")
    p_reset.add_argument("--scope", default="repo", choices=["repo", "global", "both"],
                         help="Which scope to reset.")
    p_reset.add_argument("--memory", action="store_true", help="Also clear memory.md")
    p_reset.add_argument("--session", action="store_true", help="Clear rolling current_context.md")
    p_reset.add_argument("--thread", action="store_true", help="Clear last_response_id")

    args = ap.parse_args()

    # Resolve a repo_root if available. Some commands (global note/reset) can run outside a repo.
    repo_root: Optional[Path] = None
    if getattr(args, "repo", None):
        repo_root = Path(args.repo).expanduser().resolve()
        # Validate it's a git repo.
        if not (repo_root / ".git").exists():
            # Some repos may use worktrees/submodules, so fall back to `git -C`.
            try:
                subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"], text=True)
            except Exception:
                raise SystemExit(f"--repo does not look like a git repo: {repo_root}")
    else:
        if in_git_repo():
            repo_root = git_root()

    # Commands that need a repo must enforce it.
    if args.cmd in ("plan", "patch") and repo_root is None:
        raise SystemExit("Not inside a git repo. cd into a mod repo, or pass --repo <path>.")

    global_state_dir = ensure_dir(get_global_state_dir())

    # Handle non-AI commands first (note/reset), then exit.
    if args.cmd == "note":
        scope = args.scope
        ts = datetime.utcnow().isoformat()
        note_text = f"### {ts}\n{args.text.strip()}\n"
        if scope == "global":
            target = global_state_dir / "memory.md"
            append_and_cap(target, note_text)
            print(f"[Appended note to GLOBAL memory: {target}]")
            return
        # repo
        if repo_root is None:
            raise SystemExit("Repo scope note requires being inside a git repo or passing --repo <path>.")
        repo_state = ensure_state_dir(repo_root)
        target = repo_state / "memory.md"
        append_and_cap(target, note_text)
        print(f"[Appended note to REPO memory: {target}]")
        return

    if args.cmd == "reset":
        scope = args.scope
        scopes = [scope] if scope in ("repo", "global") else ["repo", "global"]
        for s in scopes:
            base = global_state_dir if s == "global" else (ensure_state_dir(repo_root) if repo_root else None)
            if base is None:
                continue
            if args.memory:
                try:
                    (base / "memory.md").unlink(missing_ok=True)
                except Exception:
                    pass
            if args.session:
                try:
                    (base / "current_context.md").unlink(missing_ok=True)
                except Exception:
                    pass
            if args.thread:
                try:
                    (base / "last_response_id.txt").unlink(missing_ok=True)
                except Exception:
                    pass
        print("[Reset completed]")
        return

    # Load environment variables from a .env at repo root (if present)
    try:
        load_dotenv_file(repo_root)
    except Exception:
        pass

    state_dir = ensure_state_dir(repo_root)

    # Optional resets for conversation/session state.
    if getattr(args, "reset_session", False):
        if args.session_scope in ("repo", "both"):
            try:
                (state_dir / "current_context.md").unlink(missing_ok=True)
            except Exception:
                pass
        if args.session_scope in ("global", "both"):
            try:
                (global_state_dir / "current_context.md").unlink(missing_ok=True)
            except Exception:
                pass

    if getattr(args, "reset_thread", False) and args.thread_scope != "none":
        base = global_state_dir if args.thread_scope == "global" else state_dir
        try:
            (base / "last_response_id.txt").unlink(missing_ok=True)
        except Exception:
            pass

    excludes = [] if args.no_default_excludes else list(DEFAULT_EXCLUDES)
    forbidden = list(DEFAULT_FORBIDDEN_PATTERNS)

    budget = Budget(
        max_files=args.max_files,
        max_file_chars=args.max_file_chars,
        max_total_chars=args.max_total_chars,
        max_grep_chars=args.max_grep_chars,
    )

    # Select files under budget
    files, omit_notes = collect_files(
        repo_root=repo_root,
        globs=args.files,
        include_changed=args.include_changed,
        auto=args.auto,
        excludes=excludes,
        budget=budget,
    )

    # Optionally include some default important files. These often carry metadata
    # or configuration that helps the model understand the project. We only add
    # them if they exist and avoid duplicates.
    if getattr(args, "include_default", False):
        default_candidates = [
            Path("README.md"),
            Path("settings.gradle"),
            Path("src/main/resources/manifest.json"),
        ]
        for rel in default_candidates:
            p = repo_root / rel
            if p.exists():
                rel_str = str(rel)
                if rel_str not in files:
                    files.append(rel_str)

    if args.cmd == "plan":
        ctx, ctx_notes = build_context(
            repo_root=repo_root,
            mode=args.mode,
            request_text=args.prompt,
            plan_text=None,
            files=files,
            grep_queries=args.grep,
            include_full_diff=args.diff,
            budget=budget,
        )

        if args.save_context:
            (state_dir / "context.txt").write_text(ctx, encoding="utf-8")

        system = (
            "You are a senior engineer helping with a Hytale Java/Gradle mod repo.\n"
            "Output MUST be a step-by-step PLAN only.\n"
            "Rules:\n"
            "1) Produce a numbered plan with clear steps.\n"
            "2) List which files you intend to change and why.\n"
            "3) Do not output any code patches/diffs yet.\n"
            "4) Do NOT suggest changing docker/gradle config.\n"
        )

        mem = memory_blob(repo_root, args.memory_scope, args.session_scope)
        memory_section = f"MEMORY:\n{mem}\n\n" if mem else ""

        user = (
            f"MODE: {args.mode}\n\n"
            f"REQUEST:\n{args.prompt}\n\n"
            f"CONTEXT NOTES:\n- " + "\n- ".join(omit_notes + ctx_notes) + "\n\n"
            + memory_section +
            f"REPO CONTEXT:\n{ctx}\n"
        )

        # Approximate token count and enforce --max-tokens if provided.
        if args.max_tokens:
            approx_tokens = estimate_tokens(system + user)
            if approx_tokens > args.max_tokens:
                raise SystemExit(
                    f"Context too large (~{approx_tokens} tokens) exceeding --max-tokens={args.max_tokens}. "
                    "Try reducing included files or grep queries."
                )
        # Thread continuity via previous_response_id (optional)
        previous_id = None
        if args.thread_scope != "none" and not args.reset_thread:
            thread_base = global_state_dir if args.thread_scope == "global" else state_dir
            previous_id = safe_read_text(thread_base / "last_response_id.txt", max_chars=200).strip() or None

        plan, rid = call_openai(args.model, args.effort, system, user, previous_response_id=previous_id)

        # Save thread id
        if args.thread_scope != "none" and rid:
            thread_base = global_state_dir if args.thread_scope == "global" else state_dir
            (thread_base / "last_response_id.txt").write_text(rid, encoding="utf-8")

        # Update rolling session context
        ts = datetime.utcnow().isoformat()
        session_entry = (
            f"## {ts} plan\n"
            f"PROMPT: {args.prompt.strip()}\n\n"
            f"PLAN:\n{plan.strip()}\n"
        )
        if args.session_scope in ("repo", "both"):
            append_and_cap(state_dir / "current_context.md", session_entry)
        if args.session_scope in ("global", "both"):
            append_and_cap(global_state_dir / "current_context.md", session_entry)
        save_last_plan(repo_root, plan)

        payload = {
            "ts": datetime.utcnow().isoformat(),
            "mode": args.mode,
            "model": args.model,
            "effort": args.effort,
            "prompt": args.prompt,
            "repo_root": str(repo_root),
            "files": files,
            "grep": args.grep,
            "diff_included": bool(args.diff),
            "budget": budget.__dict__,
            "notes": omit_notes + ctx_notes,
        }
        save_last_request(repo_root, payload)
        write_history(repo_root, {"type": "plan", **payload})

        print(plan)
        return

    if args.cmd == "patch":
        # Load last plan/request
        try:
            last = load_last_request(repo_root)
            plan_text = load_last_plan(repo_root)
        except Exception:
            raise SystemExit("No previous plan found. Run: rr_assist.py plan ...")

        # Respect last request unless user overrides flags now
        mode = last.get("mode", args.mode)
        request_text = last.get("prompt", "")
        include_full_diff = bool(args.diff or last.get("diff_included", False))
        grep_queries = args.grep or last.get("grep", [])

        ctx, ctx_notes = build_context(
            repo_root=repo_root,
            mode=mode,
            request_text=request_text,
            plan_text=plan_text,
            files=files,
            grep_queries=grep_queries,
            include_full_diff=include_full_diff,
            budget=budget,
        )

        if args.save_context:
            (state_dir / "context.txt").write_text(ctx, encoding="utf-8")

        system = (
            "You are a senior engineer producing a unified diff patch for a git repo.\n"
            "Output MUST be a unified diff ONLY (no markdown, no commentary).\n"
            "Rules:\n"
            "1) Output only the diff. Start with 'diff --git ...'.\n"
            "2) Keep changes minimal and focused.\n"
            "3) Do NOT modify docker/gradle config files.\n"
        )

        mem = memory_blob(repo_root, args.memory_scope, args.session_scope)
        memory_section = f"MEMORY:\n{mem}\n\n" if mem else ""

        user = (
            f"MODE: {mode}\n\n"
            f"REQUEST:\n{request_text}\n\n"
            f"CONTEXT NOTES:\n- " + "\n- ".join(omit_notes + ctx_notes) + "\n\n"
            + memory_section +
            f"REPO CONTEXT (includes plan):\n{ctx}\n"
        )

        # Enforce approximate token limit if set. Include system and user messages.
        if args.max_tokens:
            approx_tokens = estimate_tokens(system + user)
            if approx_tokens > args.max_tokens:
                raise SystemExit(
                    f"Context too large (~{approx_tokens} tokens) exceeding --max-tokens={args.max_tokens}. "
                    "Try reducing included files or grep queries."
                )
        # Thread continuity via previous_response_id (optional)
        previous_id = None
        if args.thread_scope != "none" and not args.reset_thread:
            thread_base = global_state_dir if args.thread_scope == "global" else state_dir
            previous_id = safe_read_text(thread_base / "last_response_id.txt", max_chars=200).strip() or None

        raw, rid = call_openai(args.model, args.effort, system, user, previous_response_id=previous_id)

        # Save thread id
        if args.thread_scope != "none" and rid:
            thread_base = global_state_dir if args.thread_scope == "global" else state_dir
            (thread_base / "last_response_id.txt").write_text(rid, encoding="utf-8")
        diff = extract_unified_diff(raw)
        if not diff:
            raise SystemExit("Model did not return a unified diff starting with 'diff --git'. Try again or increase context.")

        touched = touched_paths_from_diff(diff)

        if not args.allow_forbidden:
            bad = [p for p in touched if path_matches_any(p, forbidden)]
            if bad:
                raise SystemExit(
                    "Refusing patch: touches forbidden files:\n"
                    + "\n".join(f"- {b}" for b in bad)
                    + "\n(Use --allow-forbidden to override.)"
                )

        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(diff, encoding="utf-8")

        write_history(repo_root, {
            "type": "patch",
            "ts": datetime.utcnow().isoformat(),
            "out": str(out_path),
            "touched": touched,
            "model": args.model,
            "effort": args.effort,
            "budget": budget.__dict__,
        })

        # Update rolling session context (store high-level info, not the whole diff)
        ts = datetime.utcnow().isoformat()
        session_entry = (
            f"## {ts} patch\n"
            f"REQUEST: {request_text.strip()}\n"
            f"TOUCHED: {', '.join(touched)}\n"
            f"PATCH_FILE: {out_path}\n"
        )
        if args.session_scope in ("repo", "both"):
            append_and_cap(state_dir / "current_context.md", session_entry)
        if args.session_scope in ("global", "both"):
            append_and_cap(global_state_dir / "current_context.md", session_entry)

        print(diff)
        print(f"\n[Saved patch to {out_path}]")

        # If multiple attempts are requested, try applying the patch and building.
        # On failure, we capture the build log and request a refined patch from the model.
        max_attempts = getattr(args, "attempts", 1)
        attempt = 1
        while max_attempts and attempt < max_attempts:
            success = False
            try:
                # Ensure working tree is clean by resetting any previous partial changes.
                sh(["git", "reset", "--hard"], cwd=repo_root, check=False)
                # Check and apply the patch
                sh(["git", "apply", "--check", str(out_path)], cwd=repo_root, check=True)
                sh(["git", "apply", str(out_path)], cwd=repo_root, check=True)
                # Run the build to validate the patch. Use build here; customize if tests needed.
                sh(["./gradlew", "build"], cwd=repo_root, check=True)
                success = True
            except Exception as e:
                # Capture stderr or message from the build failure for context
                error_log = str(e)
                # Reset working tree to discard failed changes
                sh(["git", "reset", "--hard"], cwd=repo_root, check=False)
                # Build a new user prompt incorporating the build failure log
                failure_notes = omit_notes + ctx_notes + [f"Build attempt {attempt} failed: {error_log[:5000]}"]
                refined_user = (
                    f"MODE: {mode}\n\n"
                    f"REQUEST:\n{request_text}\n\n"
                    f"CONTEXT NOTES:\n- " + "\n- ".join(failure_notes) + "\n\n"
                    + memory_section +
                    f"REPO CONTEXT (includes plan):\n{ctx}\n"
                )
                # Enforce token limit again if necessary
                if args.max_tokens:
                    approx_tokens = estimate_tokens(system + refined_user)
                    if approx_tokens > args.max_tokens:
                        raise SystemExit(
                            f"Context too large (~{approx_tokens} tokens) in refinement exceeding --max-tokens={args.max_tokens}."
                        )
                # Ask the model to produce a refined diff given the build failure
                raw_refined, rid2 = call_openai(
                    args.model,
                    args.effort,
                    system,
                    refined_user,
                    previous_response_id=(rid or previous_id),
                )
                if rid2:
                    rid = rid2
                    if args.thread_scope != "none":
                        thread_base = global_state_dir if args.thread_scope == "global" else state_dir
                        (thread_base / "last_response_id.txt").write_text(rid, encoding="utf-8")

                new_diff = extract_unified_diff(raw_refined)
                if not new_diff:
                    raise SystemExit("Model did not return a unified diff on refinement.")
                # Save the refined diff, overwrite previous file
                out_path.write_text(new_diff, encoding="utf-8")
                diff = new_diff
                print(f"\n[Refined patch attempt {attempt} saved to {out_path}]")
                attempt += 1
                continue
            # If build succeeded, we break out
            if success:
                print(f"\nBuild succeeded after applying patch (attempt {attempt}).")
                break
        return


if __name__ == "__main__":
    main()
