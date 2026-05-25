# MarketPulse

Multi-task financial forecasting platform. Project under construction — see [solution design](docs/MarketPulse_Solution_Design_v1.docx) for the full plan.

## Testing & Pre-Push Workflow

The test suite has two tiers:

- **Fast tests** (~105 tests, ~17s) run on every default invocation.
  These cover unit-level behavior: feature engineering, walk-forward
  splitter, metrics, individual Head-1 models, fetchers, cache.

- **Slow tests** (~1 test, ~30s) are end-to-end harness smokes that
  exercise production hyperparameters on real cached data (e.g. LSTM
  at `epochs=50, mc_samples=100` across 3 walk-forward folds on AAPL).
  They are marked `@pytest.mark.slow` and **excluded from default runs**.

### Running tests

```bash
# Fast suite (default — what CI runs):
pytest tests/

# Slow suite only (manual validation):
pytest tests/ -m slow

# Both:
pytest tests/ -m "slow or not slow"
```

### Pre-push hook (one-time per clone)

This repo ships a `.githooks/pre-push` hook that runs the slow suite
before every `git push` and aborts the push on failure. To activate
it after cloning, run **once**:

```bash
git config core.hooksPath .githooks
```

The config setting is local to your clone and not committed.

To bypass the hook on a specific push (e.g. for WIP branches):

```bash
git push --no-verify
```

### Why this split exists

Slow tests use realistic data sizes and production hyperparameters,
so they catch integration regressions that unit tests miss — but they
are too expensive to run on every commit. Excluding them from default
`pytest tests/` keeps the inner-loop fast; gating them at `git push`
ensures they never silently rot.
