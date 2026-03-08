#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


OLD_WAVELENGTHS = [450.0, 550.0, 650.0, 450.0, 550.0, 650.0, 720.0, 750.0, 800.0, 850.0]
NEW_WAVELENGTHS = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 720.0, 750.0, 800.0, 850.0]


WAVELENGTH_BLOCK_RE = re.compile(
    r"(?im)(?P<prefix>^[ \t]*wavelength\s*=\s*)\{(?P<body>[^}]*)\}(?P<suffix>[^\r\n]*)"
)
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class FileResult:
    path: Path
    changed: bool
    reason: str


def _format_wavelengths(values: list[float]) -> str:
    return " , ".join(f"{v:.1f}" for v in values)


def _float_list_matches(values: list[float], expected: list[float], tol: float = 1e-6) -> bool:
    if len(values) != len(expected):
        return False
    return all(abs(a - b) <= tol for a, b in zip(values, expected, strict=True))


def update_hdr_text(text: str) -> tuple[str, bool, str]:
    changed_any = False
    updated_blocks = 0
    blocks_seen = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed_any, updated_blocks, blocks_seen
        blocks_seen += 1

        body = match.group("body")
        raw_numbers = FLOAT_RE.findall(body)
        values = [float(x) for x in raw_numbers]

        if not _float_list_matches(values, OLD_WAVELENGTHS):
            return match.group(0)

        changed_any = True
        updated_blocks += 1

        prefix = match.group("prefix")
        suffix = match.group("suffix")
        return f"{prefix}{{ {_format_wavelengths(NEW_WAVELENGTHS)} }}{suffix}"

    new_text = WAVELENGTH_BLOCK_RE.sub(repl, text)
    if not changed_any:
        if "wavelength" not in text.lower():
            return text, False, "no-wavelength-key"
        if blocks_seen == 0:
            return text, False, "wavelength-format-not-matched"
        return text, False, "wavelengths-not-matching-old"
    return new_text, True, f"updated-blocks={updated_blocks}"


def update_hdr_file(path: Path, *, dry_run: bool, backup: bool) -> FileResult:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        original = path.read_text(encoding="utf-8", errors="replace")

    updated, changed, reason = update_hdr_text(original)
    if not changed:
        return FileResult(path=path, changed=False, reason=reason)

    if dry_run:
        return FileResult(path=path, changed=True, reason=f"{reason} (dry-run)")

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(original, encoding="utf-8")

    path.write_text(updated, encoding="utf-8")
    return FileResult(path=path, changed=True, reason=reason)


def iter_hdr_paths(root: Path, *, recursive: bool) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() != ".hdr":
            raise ValueError(f"Expected a .hdr file, got: {root}")
        return [root]

    if not root.is_dir():
        raise ValueError(f"Path not found: {root}")

    paths = root.rglob("*.hdr") if recursive else root.glob("*.hdr")
    return sorted(p for p in paths if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update ENVI .hdr wavelength list from the old 450/550/650 duplicated set to the new 150/250/350 set.",
    )
    parser.add_argument("path", type=Path, help="A .hdr file or a directory containing .hdr files.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top-level directory.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing files.")
    parser.add_argument("--backup", action="store_true", help="Write <file>.hdr.bak before modifying.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recursive = not args.no_recursive

    hdr_paths = iter_hdr_paths(args.path, recursive=recursive)
    if not hdr_paths:
        print(f"No .hdr files found under: {args.path}")
        return 2

    changed = 0
    skipped = 0
    for p in hdr_paths:
        result = update_hdr_file(p, dry_run=args.dry_run, backup=args.backup)
        if result.changed:
            changed += 1
            print(f"UPDATED: {result.path} ({result.reason})")
        else:
            skipped += 1
            print(f"SKIP:    {result.path} ({result.reason})")

    print(f"Done. updated={changed} skipped={skipped} total={len(hdr_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python tools/update_hdr_wavelengths.py  /mnt/d/Project/master-graduation-project/data/oil/train/feedback/aligned_full_tif_20251231-1744

python tools/update_hdr_wavelengths.py  /mnt/d/Project/master-graduation-project/data/oil/train/feedback/aligned_full_tif_20251231-1744
"""