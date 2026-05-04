# Portfolio Build — Conversation Handoff

## 1. Project & Position

- **Project:** MarketPulse — Multi-Task Financial Forecasting Engine, Project 3 of 7
- **Solution design:** MarketPulse_Solution_Design_v1.docx (committed at `docs/MarketPulse_Solution_Design_v1.docx`, also attach to this conversation)
- **Repo:** https://github.com/mashraf-portfolio/marketpulse
- **Conversation number:** Conv 2 — picks up after Phase 0 completed in Conv 1
- **Date of handoff:** 2026-05-03

## 2. Phases status

| Phase | Title | Status | Notes |
|---|---|---|---|
| 0 | Scaffold + Dual-Deploy Plumbing | **done** | All 8 micro-prompts (originally planned 6, ended up 6 + 2 fixup commits). CI #8 fully green: lint, test, quality-gate, docker-build all pass. Image builds in ~8 min on GitHub runner. |
| 1 | Data Layer + Feature Engineering | **pending** | Stub modules exist (`src/data/`, `src/features/`); all bodies raise `NotImplementedError("Implemented in Phase 1")`. Real implementation lands in Conv 2. |
| 2 | Walk-Forward Harness + Head 1a (ARIMA, Prophet, LSTM) | **pending** | Stubs in place: `src/validation/walk_forward.py`, `src/validation/metrics.py`, `src/models/head1_price/{arima,prophet,lstm}_model.py`. |
| 3 | Head 1b: TFT | **pending** | Requires `pip install -e ".[tft]"` first (PyTorch + pytorch-forecasting not yet installed). |
| 4 | Head 2: GARCH + LSTM-vol | **pending** | |
| 5 | Head 3: HMM + XGBoost + FastAPI service | **pending** | |
| 6 | Gradio UI | **pending** | |
| 7 | pytest suite + quality gates | **pending** | Note: tests are co-developed alongside each phase, not all at once at end. |
| 8 | Deployment, README, model card | **pending** | Includes replacing the placeholder README with the real one (committed source already exists locally as `marketpulse_readme_v0.md` outside the repo, ready to drop in). |

## 3. Last commit

```
SHA:        408b106
Branch:     main
Message:    fix(docker): add README.md placeholder and copy src/ into builder stage so pip install can resolve setuptools packages
Pushed:     yes
CI status:  green (CI run #8 — all 4 jobs passed)
```

## 4. What's working right now

Verified in Conv 1's final state:

- **Local Python environment.** Python 3.11.9 venv at `.venv/`. `pip install -e ".[dev]"` succeeds in ~10 min on Windows. PowerShell profile set to UTF-8 (file: `$PROFILE`).
- **Imports clean.** `python -c "import src; from src.serving.api import app; from src.config import load_yaml; from src.data.fetchers import DataNotAvailableError; from src.features.pipeline import FeatureSchemaMismatchError"` succeeds.
- **FastAPI stub serves.** `client.get('/health')` returns `Status: 200, Body: {'status': 'degraded', 'models_loaded': [], 'uptime_seconds': 0.0, 'build_id': 'dev'}` via TestClient. The `/health` endpoint uses `getattr(app.state, "start_time", time.time())` defensive fallback because TestClient doesn't run lifespan by default.
- **All other endpoints** (`/model/info`, `/forecast/price`, `/forecast/volatility`, `/classify/regime`) return `501 Not Implemented` until Phase 5.
- **Lint clean.** `ruff check src/ tests/` and `ruff format --check src/ tests/` both pass. Per-project rule: `N803` (`X` argument naming) is suppressed in `pyproject.toml`.
- **Pre-commit hooks installed and active.** Every `git commit` runs ruff + ruff-format + trailing-whitespace + end-of-file + yaml/toml checks + large-files (max 500KB) + merge-conflicts + private-key.
- **Pytest exits 0** despite zero tests collected — `tests/conftest.py` has a `pytest_sessionfinish` hook that converts exit code 5 → 0 during scaffold phase. **Remove this hook in Phase 1 once real tests exist.**
- **CI pipeline green.** GitHub Actions runs lint → test → quality-gate (skips gracefully — no `models/metadata/head1.json` yet) → docker-build. All 4 jobs pass on every push.
- **Docker image builds.** `docker build -t marketpulse:ci .` succeeds end-to-end on Linux runner (~8 min). Image is `~1.4GB` after layer dedup. Multi-stage with `python:3.11-slim`, libgomp1, non-root `app` user UID 1000, HEALTHCHECK against `/health`.
- **Config files load.** All 5 YAMLs (`config/tickers.yaml`, `wf.yaml`, `head1.yaml`, `head2.yaml`, `head3.yaml`) parse cleanly. Saudi tickers (`2222.SR`, `1180.SR`) load as strings (not floats — verified).
- **23 tickers in whitelist** across 8 markets: US (4), Saudi (2), Kuwait (5), Qatar (3), Egypt (3), UAE Dubai (3), Netherlands (3).
- **Repo organization.** Solution design committed at `docs/MarketPulse_Solution_Design_v1.docx`. Handoff template at `docs/handoff_template.md`. README is a placeholder (3 lines) — gets replaced in Phase 8.

## 5. What's blocked, broken, or undecided

**None blocked.** A few things to note for Phase 1 awareness:

- **Line endings (LF vs CRLF) on Windows.** Every commit prints `warning: in the working copy of '...': LF will be replaced by CRLF the next time Git touches it`. This is **expected and harmless** — `.gitattributes` declares `*.py text eol=lf` which means git stores LF, working tree shows CRLF. Both ruffs (local Windows, CI Linux) agree on the canonical content. Don't try to "fix" the warnings — they're noise, not errors.
- **`.gitignore` `.gitkeep` negation pattern is fragile.** `.gitignore` says `models/checkpoints/` followed by `!models/checkpoints/.gitkeep`, but the negation didn't auto-include the file — needed `git add -f` once. **Once the .gitkeep is tracked, future commits don't need `-f`.** Lesson logged for any future "directory with one tracked sentinel file" pattern.
- **TFT/PyTorch not yet installed locally.** The venv only has `[dev]` extras, not `[tft]`. **Install `pip install -e ".[tft]"` at the start of Phase 3** when TFT actually needs it. Skipping until then keeps venv ~600MB lighter.
- **`USE_REMOTE_API=true` in `docker-compose.yml`.** When the local Gradio container talks to the local API container, this is correct. But the Gradio app on HF Spaces deploys with `USE_REMOTE_API=false` (uses bundled checkpoints, no Railway dependency). Confirmed in solution design § 2.3.

## 6. Open ADRs / decisions deferred

- **ADR-001 to ADR-005** in `docs/adr/` are **planned but not yet written**. They get authored in Phase 8 alongside the model card. Solution design § 3 lists them all. No blockers — they're documentation tasks, not gating decisions.
- **Quality-gate floors in `config/wf.yaml`** are placeholder values (`head1_aapl_dir_acc_min: 0.52`, `head2_binary_acc_min: 0.65`, `head3_f1_macro_min: 0.60`). These get *informed* by actual walk-forward results in Phases 2, 4, 5 — adjust the floors based on what real models actually achieve, not what the spec optimistically said. **Re-tune after Phase 5.**

## 7. Files changed since last handoff

This is the entire repo state at end of Conv 1 (everything is "new" — Phase 0 was the first commits).

```
.dockerignore
.env.example
.gitattributes
.gitignore
.github/workflows/ci.yml
.github/workflows/deploy_hf.yml
.pre-commit-config.yaml
app.py                        (HF Spaces entry — 5-line stub)
config/{tickers,wf,head1,head2,head3}.yaml
data/.keep_whitelist
data/cache/.gitkeep
data/seed_cache.py            (stub — Phase 1)
docker-compose.yml
docs/MarketPulse_Solution_Design_v1.docx
docs/handoff_template.md
Dockerfile                    (multi-stage, Railway target — TF only, no PyTorch)
Dockerfile.gradio             (TF + PyTorch, local dev only — NOT YET TESTED, may need same `src/` + README copy fix the main Dockerfile got)
LICENSE                       (MIT)
models/checkpoints/.gitkeep
models/feature_names.json     ({} placeholder)
models/metadata/.gitkeep
packages.txt                  (libgomp1, libssl-dev, git-lfs)
pyproject.toml                (21 prod deps, [tft] extras, [dev] extras, ruff config, pytest config)
railway.toml                  (--port $PORT, healthcheckTimeout=30s)
README.md                     (3-line placeholder — replace in Phase 8)
README_HF.md                  (HF Space front page — Phase 8 generates the YAML-prepended copy)
scripts/{seed_hf_space,train_all,smoke_test_railway,smoke_test_hf}.py  (all stubs)
src/__init__.py
src/config.py                 (REAL — loads YAMLs, lru_cache wrapped)
src/data/{fetchers,cache,tickers}.py
src/features/{technical,lagged,calendar,regime_tags,pipeline}.py
src/models/base.py
src/models/head1_price/{arima,prophet,lstm,tft}_model.py
src/models/head2_volatility/{garch,lstm_vol}_model.py
src/models/head3_regime/{hmm,xgb_regime}_model.py
src/serving/{schemas,service,api}.py    (api.py is REAL stub — boots, /health works)
src/ui/gradio_app.py
src/validation/{walk_forward,metrics}.py
tests/__init__.py
tests/conftest.py             (sys.path injection + pytest_sessionfinish exit-5→0 hook)
```

Total commits in Conv 1: 8 commits. Final commit `408b106`.

## 8. Phase scope for THIS conversation

- **Primary:** Phase 1 — Data Layer + Feature Engineering. This is the foundation all 3 forecasting heads consume. Solution design § 4 Phase 1 is the spec. Includes:
  - Real `YFinanceFetcher.fetch()` — yfinance with `auto_adjust=True`, validates non-empty
  - `joblib.Memory` cache wrapper at `data/cache/`
  - `load_whitelist()` reads `config/tickers.yaml`
  - All 4 feature sub-modules: `technical.py` (12 indicators via `ta` library), `lagged.py`, `calendar.py`, `regime_tags.py`
  - `engineer_features(df)` master function — applies all 4 sub-modules in deterministic order
  - `feature_columns()` — writes/validates `models/feature_names.json`
  - Target generators (`y_price_1d`, `y_dir_1d`, `y_logvol_1d`, `y_high_vol_1d`, `y_regime`)
  - `data/seed_cache.py` real implementation — pre-warms cache for all 23 tickers
  - `tests/test_fetchers.py`, `tests/test_features.py` — meaningful tests (determinism, idempotency, NaN safety)

- **Secondary (if budget allows):** Phase 2 — Walk-Forward harness + ARIMA/Prophet/LSTM baselines. Lighter than Phase 1 since stubs already exist; mostly fill in `fit/predict` + tests + 3-fold CV verification on AAPL.

- **Out of scope this conversation:** Phase 3 (TFT — needs PyTorch install + ~5min/fold compute, dedicated focus). Phases 4-8.

## 9. The very first action to take

Before issuing any micro-prompt, do these three things in order:

1. **Confirm working tree is clean.** Run `git status` mentally / via the user — should show "nothing to commit, working tree clean" with `main` up to date with `origin/main` at `408b106`.
2. **Confirm venv is active.** User's prompt should read `(.venv) PS C:\Users\HP\marketpulse>`. If `(.venv)` prefix is missing, instruct: `.\.venv\Scripts\Activate.ps1`.
3. **Read solution design § 4 Phase 1 carefully**, then propose a **5-prompt decomposition for Phase 1** (suggested seams: data fetcher + cache + tickers; technical + lagged feature modules; calendar + regime_tags + pipeline orchestration; target generators + seed_cache implementation; tests). Confirm the decomposition with the user before issuing prompt 1 of 5.

## 10. Anything else the assistant needs to know

- **Environment is Windows PowerShell, NOT WSL.** Path A from Conv 1 setup. No `make`, no bash. All commands must be PowerShell-native or generic enough to work in either. The user's PowerShell profile is set to UTF-8.
- **No `Co-authored-by: Claude` tags on any commit. Ever.** Conventional commit format only (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). This is a hard rule from Conv 1.
- **Pre-commit hook will reject any commit that fails ruff or has unused vars.** When stubs become real implementations, expect F841 errors if you assign `args = parse_args()` and don't use them. Use bare `parser.parse_args()` if return value isn't needed.
- **PowerShell pagination cuts off long output.** When user pastes terminal output, suspect missing lines if the paste seems truncated. Workaround: `command | Out-Host -Paging:$false` for anything >20 lines.
- **CI runs on every push to main.** Trust it as the canonical truth — if local lint/test passes but CI fails, CI is right (Linux is the deployment target). Don't waste time arguing with CI.
- **The user's GitHub username is `mashraff2024` (commit author) but the repo org is `mashraf-portfolio`** — both are valid. Don't get confused.
- **Hard constraint: NO insurance content anywhere.** Already in system prompt; restating for safety.
- **Time budget hint:** Conv 1 took ~6 hours of real work for Phase 0 (heavy on debugging genuinely tricky issues — Docker setuptools, .gitignore negation, ruff version drift). Phase 1 should be smoother because: (a) we're past the infra setup pain, (b) the data layer is mostly straightforward Python with few external deps, (c) we've established the working pattern. Plan ~2-3 hours for Phase 1.

## END HANDOFF
