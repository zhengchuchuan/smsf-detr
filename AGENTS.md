# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: training/testing entrypoint (Hydra + OmegaConf).
- `configs/`: configuration tree (`runtime/`, `data/`, `model/`, `task/`). Add new experiments under `configs/task/<task>/`.
- `engines/`: trainers, model wrappers, logging, and config parsing.
- `datasets/`: dataset adapters and loading utilities.
- `infer/`: inference scripts (e.g. `infer/run_test.py`).
- `tools/`: one-off inspection/smoke scripts for quick validation.
- `data/`, `outputs/`, `weights/`, `third_party/`: local artifacts and vendored code (gitignored by default; don’t rely on them being present in a fresh clone).

## Build, Test, and Development Commands

- Train: `python main.py --mode train --config configs/task/msifdetr/coco_rgb-msifdetr_small-det-no_fuse.yaml`
- Evaluate: `python main.py --mode test --config outputs/<run_dir>/config.yaml`
- Override config values: `python main.py ... --opts runtime.device=cpu train.epochs=1`
- Inference (test split + visual samples): `python infer/run_test.py --config outputs/<run_dir>/config.yaml --checkpoint outputs/<run_dir>/checkpoint_best.pth`
- Export ONNX: `python export_onnx.py --config outputs/<run_dir>/config.yaml --checkpoint outputs/<run_dir>/checkpoint_best.pth --output model.onnx`
- Smoke checks: `python tools/smoke_backbone_forward.py` and `python tools/smoke_rtdetrv4_vendored.py`

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep modules focused, and prefer explicit types on public helpers/IO boundaries.
- Config naming: keep filenames lowercase; follow `configs/task/msifdetr/README.md` (dataset/model/task/fusion/comment) and reuse existing `base.yaml` defaults instead of duplicating fields.

## Testing Guidelines

- There is no dedicated unit-test suite in the root project. Validate changes with:
  - a smoke script from `tools/`, and
  - a short run (e.g. `--opts train.epochs=1`) plus one `--mode test` pass.

## Commit & Pull Request Guidelines

- Match the existing Conventional Commits style: `feat(scope): ...`, `fix(scope): ...`, `refactor(scope): ...` (scopes seen: `infer`, `export_onnx`, `rtmsfdetr`).
- PRs should include: the config used, how to reproduce (exact command), and key outputs (metrics/log excerpts; screenshots when changing visualization or inference rendering).

## Security & Configuration Tips

- Never commit secrets (API keys, tokens). Use environment variables and local `.env` files.
- Keep large artifacts out of git (datasets, `*.pth`/`*.pt`, `*.tif`); place them under gitignored paths like `data/`, `outputs/`, and `weights/`.
