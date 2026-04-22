# Changelog

All notable changes to this project are documented here, one PR per entry,
newest first, dated by merge to `main`. Backfilled from git history.

## 2026-04-21 — PR #34

**Breaking — minimum Python bumped from `>=3.10` to `>=3.11`.** Required by
`cuml-cu13==26.4`. Also updates `target-version` in the black config and
`CONTRIBUTING.md`.

## 2026-04-20 — PR #33

Switch `cvar_basic.ipynb` to render the architecture diagram via PNG to
avoid SVG rendering issues in `nbconvert` output.

## 2026-04-20 — PR #32

Large bundled change:

- **Added** — Mean-Variance (Markowitz) optimizer with CVXPY and cuOpt
  backends; `notebooks/mean_variance_basic.ipynb`.
- **Added** — `BaseOptimizer` / `BaseParameters` classes and Pydantic
  settings models (`ReturnsComputeSettings`, `ScenarioGenerationSettings`,
  `KDESettings`, `ApiSettings`) under `cufolio.settings`.
- **Added** — SP100, DOW30, and Global Titans dataset download support in
  `utils.download_data`.
- **Added** — `compute_absolute_returns` (simple-diff semantics) as a
  distinct function.
- **Breaking — API rename** — `compute_abs_returns` → `compute_linear_returns`
  (pct-change). Previous `diff` semantics now live in `compute_absolute_returns`.
- **Breaking — settings API** — `calculate_returns`, `generate_cvar_data`,
  and optimizer constructors now take Pydantic settings models instead of
  plain dicts.
- **Breaking — dependency pins** — `cuml-cu{12,13}` and `cuopt-cu{12,13}`
  bumped from `25.12.*` to `26.4.*` / `26.04.*`; `pydantic>=2.12.5` added.
- **Changed** — `optimize_market_regimes()` unified to support both CVaR
  and mean-variance.
- **Fixed** — cuML KDE sampler returns numpy arrays via `using_output_type`.
- **Fixed** — `quad_form` numerical issues resolved via `psd_wrap` on the
  covariance matrix.
- Notebooks rewritten against the new API; outputs regenerated.

## 2025-12-17 — PR #1 (from fork)

- **Changed** — cuOpt model construction switched to `LinearExpression` to
  avoid recursion depth errors on large problems.
- **Changed** — refactored returns-calculation methods; made exception
  handling explicit.
- **Added** — KDE timing instrumentation in the CVaR pipeline.
- **Fixed** — lazy imports in `cvar_optimizer` to avoid eager GPU-module
  load at import time.

## 2025-12-02 — PR #28

Small touch-up of `notebooks/launchable.ipynb` (1 file, +161/-30).

## 2025-12-02 — PR #27

Minor launchable-notebook fix (4 files, +33/-10).

## 2025-12-02 — PR #26

Major launchable-notebook iteration: GPU checks, setup steps, regenerated
outputs (7 files, +3095/-173).

## 2025-12-01 — PR #25

Rename / replace the architecture diagram asset (`docs/arch_diagram.png`).

## 2025-12-01 — PR #24

Refresh notebook outputs to match latest src behavior (4 files, +213/-171).

## 2025-11-26 — PR #23

Final pass on the stock-data directory setup (4 files, +1596/-1813) —
regenerated dataset artifacts and README polish.

## 2025-11-25 — PR #22

Add a bundled dataset file (1 file, +432/-12).

## 2025-11-25 — PR #21

Polish on the stock-data directory layout (3 files, +15/-2).

## 2025-11-25 — PR #20

Initial `data/stock_data/` scaffolding (5 files, +51/-68).

## 2025-11-25 — PR #19

Small source touch-up (1 file, +7/-6).

## 2025-11-25 — PR #18

Launchable-notebook refinements (7 files, +1187/-1326).

## 2025-11-24 — PR #17

Add Brev-deployable `notebooks/launchable.ipynb` (9 files, +4855/-747).

## 2025-11-24 — PR #16

Finalize `uv.lock` migration and sequence diagram (5 files, +1377/-1091).

## 2025-11-23 — PR #15

Refine `uv.lock` sequence diagram docs (6 files, +67/-49).

## 2025-11-21 — PR #14

Commit `uv.lock` and migrate project to `uv` as the dependency manager;
add a sequence diagram documenting the dependency-management flow
(5 files, +2928/-6).

## 2025-11-21 — PR #13

Add `release-2512.yml` workflow, `docs/arch_diagram.png`, large notebook
regeneration, and `src/` cleanup (11 files, +1505/-2082).

## 2025-11-20 — PR #12

Small notebook fixes in `cvar_basic` and `rebalancing_strategies`
(2 files, +4/-4).

## 2025-11-19 — PR #11

Notebook regeneration and src cleanup in backtest / rebalance / utils
(5 files, +703/-912).

## 2025-11-19 — PR #10

Notebook and src refinements across cvar_basic, backtest, cvar_utils,
portfolio, rebalance, utils (7 files, +304/-141).

## 2025-11-17 — PR #9

README additions (+36 lines).

## 2025-11-13 — PR #8

`src/readme.md` fix (1 file, +2/-2).

## 2025-11-12 — PR #7

Remove `deploy/1_Deploy_Template.ipynb`; major `cvar_basic.ipynb` update;
add `notebooks/readme.md`; minor `src/cvar_utils.py` and `src/utils.py`
tweaks (7 files, +1358/-1235).

## 2025-11-12 — PR #6

README polish (1 file, +18/-12).

## 2025-11-12 — PR #5

Remove legacy scaffolding: `Maintainers/`, old `CHANGELOG.md`,
`README_ARCHIVED.md`, empty `customize/` and `evaluate/` readmes; README
polish (10 files, +74/-409).

## 2025-11-07 — PR #4

`LICENSE-3rd-party.txt` edits, remove a `pyproject.toml` entry
(2 files, +5/-7).

## 2025-11-06 — PR #3

Add `LICENSE-3rd-party.txt`; README touch-up (2 files, +119/-2).

## 2025-11-06 — PR #2

README polish (1 file, +2).

## 2025-11-06 — PR #1

**Initial code drop.** Core `cufolio` package: CVaR optimizer
(`cufolio.cvar_optimizer.CVaR`), portfolio backtester
(`cufolio.backtest.portfolio_backtester`), utils, scenario generation
(KDE / Gaussian / historical). Example notebooks: `cvar_basic.ipynb`,
`efficient_frontier.ipynb`, `rebalancing_strategies.ipynb`. `pyproject.toml`
with `cuda12` / `cuda13` extras pinning `cuml-cu{12,13}` and
`cuopt-cu{12,13}`. `README.md`, `CONTRIBUTING.md`, `CODEOWNERS`,
NVIDIA-AI-Blueprints branding and issue templates (22 files, +10885/-25).
