#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC_IMG_ROOT="$ROOT/data/oil_20260202/rgb"
SRC_LBL_ROOT="$ROOT/data/oil_20260202/s2adet_msi_20260202_3cls/rgb/labels"
DST_ROOT="$ROOT/data/oil_20260202/ultralytics_rgb_20260202_3cls"

mkdir -p "$DST_ROOT/images" "$DST_ROOT/labels"

for split in train val test; do
  if [[ ! -d "$SRC_IMG_ROOT/$split" ]]; then
    echo "Missing images split dir: $SRC_IMG_ROOT/$split" 1>&2
    exit 2
  fi
  if [[ ! -d "$SRC_LBL_ROOT/$split" ]]; then
    echo "Missing labels split dir: $SRC_LBL_ROOT/$split" 1>&2
    exit 2
  fi

  # Important: Ultralytics derives label paths by replacing '/images/' with '/labels/' in image paths.
  # If we symlink the whole images/ dir to an external folder (e.g. .../rgb/train), that '/images/' segment
  # disappears after resolving, and Ultralytics can't find labels. So we create per-file symlinks.
  mkdir -p "$DST_ROOT/images/$split" "$DST_ROOT/labels/$split"

  # Images
  for img in "$SRC_IMG_ROOT/$split"/*; do
    [[ -f "$img" ]] || continue
    bn="$(basename "$img")"
    rel="$(realpath --relative-to="$DST_ROOT/images/$split" "$img")"
    dst="$DST_ROOT/images/$split/$bn"
    if [[ ! -e "$dst" || -L "$dst" ]]; then
      ln -snf "$rel" "$dst"
    fi
  done

  # Labels
  for lbl in "$SRC_LBL_ROOT/$split"/*.txt; do
    [[ -f "$lbl" ]] || continue
    bn="$(basename "$lbl")"
    rel="$(realpath --relative-to="$DST_ROOT/labels/$split" "$lbl")"
    dst="$DST_ROOT/labels/$split/$bn"
    if [[ ! -e "$dst" || -L "$dst" ]]; then
      ln -snf "$rel" "$dst"
    fi
  done
done

echo "Prepared Ultralytics dataset wrapper at: $DST_ROOT"
