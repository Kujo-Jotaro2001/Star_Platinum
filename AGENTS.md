## Project goal

ML-driven trading bot for Bybit USDT-margined perpetual futures.

Current status:

* Phase 1 ingestion: complete.
* Phase 2 offline feature pipeline: complete.
* Phase 3 model + training: complete.
* Current work: Phase 4 signals/risk and Phase 5 inference/execution/backtest.

MVP scope:

* Single symbol: BTCUSDT.
* Bybit Demo by default.
* Main objective now: turn trained model outputs into safe, testable trading decisions.

## Repository layout

```text
bot/
  data/          # Phase 1 ingestion and historical loading
  features/      # Phase 2 offline feature extraction
  models/        # Phase 3 model architecture
  training/      # Phase 3 training loop, losses, metrics
  signals/       # Phase 4 signal generation from model outputs
  risk/          # Phase 4 risk checks and position/order constraints
  inference/     # Phase 5 online feature prep and model prediction
  execution/     # Phase 5 Bybit order placement/cancel/position handling
  backtest/      # Phase 5 offline strategy simulation

conf/
  config.yaml
  ingestion/default.yaml
  features/default.yaml
  model/default.yaml
  train/default.yaml
  historical/default.yaml

tests/
  data/
  features/
  training/
```

## Stack and core constraints

* Exchange: Bybit v5 API, pybit v5.
* Use pybit directly. Do not create proxy/wrapper classes around the SDK unless explicitly requested.
* Async runtime: asyncio + uvloop.
* Storage: SQLite daily rotation, aiosqlite for async writes.
* Config: Hydra + OmegaConf, plain YAML only.
* ML: PyTorch + PyTorch Lightning.
* Secrets: environment variables only, optionally loaded with python-dotenv. Never put secrets in YAML.

## Configuration rules

All tuneable parameters must live in `conf/`.

Rules:

* Use Hydra entrypoints with `@hydra.main(..., version_base=None)`.
* Entrypoint `main(cfg: DictConfig)` extracts config values.
* Inner functions/classes take plain typed arguments, not `cfg`.
* Inner functions must not import or reference global config.
* No typed config dataclasses / Structured Configs.
* CLI overrides must remain supported, for example:

```bash
python -m bot.features.pipeline features.start_date=2026-04-12
```

`cfg.ingestion.lob_depth` is the single source of truth for LOB depth `K`. If `cfg.model.ob_depth` exists, it must match `cfg.ingestion.lob_depth`.

## Existing data/model contracts

Phase 2 output files under `data/features/{SYMBOL}/`:

```text
ob_raw.npy             [N, K, 4] float32
flow_features.npy      [N, 9]    float32
ctx_features.npy       [N, 17]   float32
labels.npy             [N, H]    int8
flat_mask.npy          [N, H]    bool
timestamps_ms.npy      [N]       int64
is_reset.npy           [N]       bool
normalizer_stats.npz
```

Important contracts:

* LOB tensor is self-normalizing: price offsets + log1p sizes.
* Do not z-score LOB features.
* Trade flow features use rolling z-score.
* Flow normalizer is fit on train split only and reused online via `normalizer_stats.npz`.
* Context features are forward-filled and not normalized.
* Labels are multi-horizon classes: `0=down`, `1=flat`, `2=up`.
* Flat samples are kept in labels and masked in training loss.

Model inference contract:

* Model input windows:

  * `ob`: `[B, T, K, 4]`
  * `flow`: `[B, T, 9]`
  * `ctx`: `[B, T, 17]`
* Model output:

  * logits `[B, H, C]`
  * `H = len(cfg.features.label_horizons)`
  * `C = 3`

Do not change these shapes without updating dataset, training, inference, and tests together.

## Online vs offline parity

The same trained model must be used offline and online.

Offline training path:

* `ob_raw.npy` → `LOBDataset` window → model
* `flow_features.npy` → `LOBDataset` window → model
* `ctx_features.npy` → `LOBDataset` window → model

Online inference path:

* WS order book snapshot → `serialize_ob_snapshot` → `RingBuffer[T,K,4]`
* WS trades → trade-flow extractor → `Normalizer.normalize_only()` → `RingBuffer[T,9]`
* WS ticker/liquidations/long-short ratio → context cache → `RingBuffer[T,17]`

Phase 5 must preserve this parity. Do not implement online-only feature definitions that differ from offline training features.

## Phase 4 design rules: signals and risk

Model logits are not trading decisions.

The `signals/` layer is responsible for converting model outputs into typed trading signals. A signal should include at least:

* symbol
* timestamp
* side/action: buy, sell, hold, close, or no-trade
* confidence/probability
* selected horizon
* optional reason/debug fields

The `risk/` layer is a mandatory gate before execution.

Execution must never bypass risk checks.

Risk checks should be deterministic and testable. Typical checks:

* max position size
* max notional exposure
* max leverage
* max daily loss / drawdown limit
* cooldown after loss or liquidation-like event
* minimum confidence threshold
* spread/liquidity filter
* stale data filter
* warmup/not-enough-history filter

If risk rejects a signal, execution must not send an order.

## Phase 5 design rules: inference, execution, backtest

### Inference

Online inference should:

* load the trained checkpoint
* load `normalizer_stats.npz`
* maintain sliding windows with `RingBuffer`
* wait until all buffers are warm before predicting
* return a typed prediction/signal payload, not raw tensors only
* avoid lookahead or future data leakage

### Execution

Execution should:

* default to demo/dry-run behavior unless explicitly configured otherwise
* use pybit v5 directly
* keep API keys in environment variables only
* be idempotent where possible
* avoid duplicate orders for the same signal
* log order intent and result at the entrypoint/service boundary
* separate order decision from order submission

Do not add live-mainnet trading behavior unless explicitly requested.

### Backtest

Backtest must reuse the same signal and risk logic as live trading.

Backtest should model at least:

* fees
* slippage or conservative fill assumptions
* spread
* position state
* realized/unrealized PnL
* drawdown
* rejected signals

Backtest must be chronological. Never shuffle time-series data.

## Coding conventions

* All tuneable numbers go through config.
* No hardcoded trading thresholds inside business logic.
* Prices and quantities: use `Decimal` in memory where precision matters.
* DB stores prices/quantities as strings.
* Feature/model boundary uses `float32`.
* Timestamps: UTC unix milliseconds as `int`.
* Every public function must have type hints.
* Keep business logic deterministic and easy to unit test.
* Tests must use synthetic data where possible.
* No network calls in unit tests.
* No live exchange calls in tests.

## Logging

Use `structlog`.

Logging belongs at pipeline/service entrypoints, not deep inside pure extractors, encoders, or deterministic business logic.

Good places for logs:

* ingestion entrypoint
* feature pipeline entrypoint
* training entrypoint
* inference service loop
* execution boundary
* backtest run summary

Avoid logs inside pure functions.

## Strict code style

* No proxy/wrapper classes around pybit SDK.
* No broad `try/except`.
* Use `try/except` only for a specific known exception with real recovery behavior.
* No defensive “just in case” code.
* No comments that simply restate what the code does.
* Comments should explain why something non-obvious is necessary.
* Prefer small typed functions over large stateful classes.
* Do not introduce new production dependencies without explicit approval.

## Verification expectations

When changing code, run the most relevant checks.

Preferred checks:

```bash
pytest
```

For targeted work, run the smallest relevant test subset first, for example:

```bash
pytest tests/features
pytest tests/training
```

For Phase 4/5 changes, add or update tests under:

```text
tests/signals/
tests/risk/
tests/inference/
tests/execution/
tests/backtest/
```

A task is done only when:

* the requested behavior is implemented
* relevant tests are added or updated
* relevant tests pass, or the reason they cannot be run is clearly stated
* the final response summarizes changed files and remaining risks

## Decision-making rules

Before implementing large design changes:

* inspect the existing code first
* propose a short plan
* preserve existing contracts unless explicitly asked to change them
* ask before changing architecture, data formats, model shapes, or config philosophy

For small localized fixes, implement directly and summarize the change.

If multiple approaches exist, prefer the simplest one that preserves existing contracts and is easy to test.
