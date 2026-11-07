"""Check tracked text files against locally configured public-text patterns."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def configured_patterns(root: Path) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    environment = os.environ.get("PUBLIC_CHECK_TERMS", "")
    if environment:
        sources.extend(
            ("PUBLIC_CHECK_TERMS", line) for line in environment.splitlines()
        )

    local_file = root / ".public-check-terms"
    if local_file.is_file():
        sources.extend(
            (".public-check-terms", line)
            for line in local_file.read_text(encoding="utf-8").splitlines()
        )

    patterns: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, value in sources:
        value = value.strip()
        if not value or value.startswith("#") or value in seen:
            continue
        seen.add(value)
        patterns.append((source, value))
    return patterns


def tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        cwd=root,
    )
    return [root / os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def main() -> int:
    try:
        root = repository_root()
        configured = configured_patterns(root)
        files = tracked_paths(root)
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        print(f"Could not inspect tracked files: {error}", file=sys.stderr)
        return 2

    if not configured:
        print(
            "Warning: no public-text patterns are configured; check skipped.",
            file=sys.stderr,
        )
        return 0

    compiled: list[tuple[str, re.Pattern[str]]] = []
    for source, pattern in configured:
        try:
            compiled.append((pattern, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)))
        except re.error:
            print(f"Invalid regular expression in {source}.", file=sys.stderr)
            return 2

    findings: list[str] = []
    scanned = 0
    for path in files:
        if path.name.upper() == "LICENSE":
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            print(f"Could not read {path.relative_to(root)}: {error}", file=sys.stderr)
            return 2
        if b"\0" in content[:8192]:
            continue
        scanned += 1
        text = content.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, expression in compiled:
                if expression.search(line):
                    findings.append(
                        f"{path.relative_to(root)}:{line_number}: {pattern}"
                    )

    if findings:
        print("\n".join(findings), file=sys.stderr)
        return 1

    print(f"Public-text check passed ({scanned} tracked text files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
