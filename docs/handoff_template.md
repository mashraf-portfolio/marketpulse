# Conversation Handoff Template

> **Purpose:** Bridge between consecutive Claude conversations as Mohammad Ashraf builds the 7-project portfolio. Each conversation typically holds 2-3 phases of one project before context fills. This template is the **first user message** of the next conversation — paste it in along with the relevant solution design `.docx` and any other artifacts.

---

## How to use this template

**At the END of each conversation:**
1. Copy this file.
2. Fill in every section below with the live status as of conversation end.
3. Save the filled-in copy as a new file (e.g. `handoffs/mp_conv1_to_conv2.md`) in the project repo so the history survives.
4. Open a new Claude conversation in the same Claude Project.
5. Attach the relevant solution design `.docx` (e.g. `MarketPulse_Solution_Design_v1.docx`).
6. Paste the filled-in template as the first message.

**The Portfolio Building Assistant should respond by:** confirming it understands the state, reviewing the solution design for the upcoming phase, decomposing the next phase into 3-6 micro-prompts, and asking for the go-ahead to start the first one.

---

## Filled-in template starts below the line — copy from `## START HANDOFF` downward

---

## START HANDOFF

# Portfolio Build — Conversation Handoff

## 1. Project & Position

- **Project:** _(e.g. MarketPulse — Multi-Task Financial Forecasting Engine, Project 3 of 7)_
- **Solution design:** _(filename, attached to this conversation, e.g. MarketPulse_Solution_Design_v1.docx)_
- **Repo:** _(e.g. github.com/mashraf-portfolio/marketpulse)_
- **Conversation number:** _(e.g. Conv 2 — picks up where Conv 1 stopped)_
- **Date of handoff:** _(YYYY-MM-DD)_

## 2. Phases status

Mark each phase: `done` / `in-progress` / `pending` / `skipped (with reason)`.

| Phase | Title | Status | Notes |
|---|---|---|---|
| 0 | Scaffold + Dual-Deploy Plumbing | _e.g. done_ | _e.g. CI green, Docker builds in 7 min_ |
| 1 | Data Layer + Feature Engineering | _in-progress_ | _80% — engineer_features works, missing regime_tags tests_ |
| 2 | _phase title_ | _pending_ | |
| 3 | _phase title_ | _pending_ | |
| _..._ | | | |

## 3. Last commit

```
SHA:        _(7-char prefix, e.g. a1b2c3d)_
Branch:     _(usually main)_
Message:    _(verbatim commit message)_
Pushed:     _(yes/no — if no, flag here)_
CI status:  _(green / red / pending — link to Actions run if red)_
```

## 4. What's working right now

Bullet list of concrete capabilities verified in the last session. Be specific — "tests pass" is too vague; "all 14 tests in tests/test_features.py pass with coverage 87%" is right.

- _e.g. `make seed-cache` fetches and caches all 23 tickers, 2.8 min cold run_
- _e.g. `engineer_features(df)` produces 41 columns, deterministic, idempotent_
- _e.g. WalkForwardSplitter passes all 4 disjoint-index tests_

## 5. What's blocked, broken, or undecided

If nothing is blocked, write `none`. Otherwise be explicit — what's blocked, on what decision, and your current preference.

- _e.g. Optuna search space for XGBoost regime — undecided whether to widen `max_depth` to [3,12] given Egypt tickers' irregular distributions. Current preference: keep [3,8], note in ADR._
- _e.g. yfinance returning 401 for ZAIN.KW intermittently — workaround: retry with 2s backoff. If persists past Phase 4, switch to Alpha Vantage fallback._

## 6. Open ADRs / decisions deferred

Architecture Decision Records that were started but not closed, OR pivots that need to be ratified before the next phase.

- _e.g. ADR-002 (TF/PyTorch coexistence) — marked Proposed, awaiting confirmation that Railway image actually OOMs at ~3GB before locking it as Accepted._
- _e.g. None._

## 7. Files changed since last handoff

Rough list — doesn't have to be exact. The next conversation will `git diff` if needed, but a list helps it know where to look.

```
src/data/fetchers.py            (new)
src/data/cache.py               (new)
src/features/pipeline.py        (modified — added regime_tags wiring)
config/wf.yaml                  (modified — bumped max_folds 25→30)
tests/test_features.py          (new — 9 tests)
data/cache/AAPL.parquet         (new, LFS)
... etc
```

## 8. Phase scope for THIS conversation

Which phase(s) will this new conversation tackle. Be realistic — 2 phases typical, 3 only if they're light.

- **Primary:** _e.g. Phase 2 — Walk-Forward Validation Harness + Head 1a (ARIMA, Prophet, LSTM)_
- **Secondary (if budget allows):** _e.g. Phase 3 — Head 1b TFT — kickoff only, fitting may run overnight outside the conversation_
- **Out of scope this conversation:** _e.g. Phase 4 onwards. Phase 7 tests are co-developed alongside each phase but the dedicated test-pass happens in a later conversation._

## 9. The very first action to take

What I (the assistant) should do as my first move when responding to this handoff. One sentence.

- _e.g. "Read §4 Phase 2 of the attached solution design, then propose a 4-prompt decomposition (walk-forward harness → ARIMA → Prophet → LSTM), confirm with me before issuing the first."_
- _e.g. "Run `git status` against the live repo and confirm the working tree is clean before we begin."_

## 10. Anything else the assistant needs to know

Free-form. Constraints, time pressure, deadline, recent pivots, things the previous conversation discovered the hard way.

- _e.g. "Today I have ~2 hours. Prefer getting Phase 2 fully done over starting Phase 3."_
- _e.g. "We discovered yesterday that BOURSA.KW has a 6-month gap in 2021 (suspended trading). Excluded from walk-forward subset."_
- _e.g. "No insurance content anywhere — hard rule from system prompt, restating for safety."_

## END HANDOFF

---

## Mini-version — for short handoffs (single phase remaining, no pivots)

If the conversation ends mid-phase with nothing complicated, this minimal version is enough:

```
# Quick Handoff — <Project> Phase <N>

Last commit: <sha> "<message>"
Status:      Phase <N> at <X>% — <one-line summary>
Next move:   <one sentence>
Blocked on:  <none / specifics>
```

---

## What this template intentionally does NOT include

- **System prompt** — already in the Claude Project; never repeat it.
- **Full project blueprint** — already in the system prompt and the solution design.
- **Tutorial / explanation of the project** — assistant has full context from the system prompt.
- **Code snippets** — the next conversation reads code from the repo, not from chat history.
