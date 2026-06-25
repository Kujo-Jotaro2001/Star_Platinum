# Trading Bot — Bybit Perpetual Futures

## Project goal
ML-driven trading bot for Bybit USDT-margined perpetual futures.
Phase 1 (ingestion) complete. Phase 2 (offline feature pipeline) complete.
Phase 3 (model + training) complete.

## Stack
- **Exchange:** Bybit v5 API, pybit v5 — used directly, no proxy classes
- **Async:** asyncio + uvloop
- **Storage:** SQLite daily rotation (`{SYMBOL}_{YYYY-MM-DD}.db`), aiosqlite for async writes
- **Config:** Hydra + OmegaConf — pure YAML under `conf/`, no typed dataclasses / Structured Configs
- **Features:** numpy .npy arrays (Phase 2 output)
- **ML:** PyTorch + PyTorch Lightning, CUDA (local GPU) — Phase 3+
- **Secrets:** environment variables only (python-dotenv), never in YAML

## Project structure
```
bot/
  main.py               # Phase 1 entrypoint, @hydra.main
  data/                 # Phase 1 — complete
    models.py           # dataclasses: OrderBookSnapshot, Trade, TickerContext, Liquidation, LongShortRatio
    orderbook.py        # thread-safe container for pybit WS book state
    storage.py          # async queue consumer, SQLite daily rotation
    ws_client.py        # DataIngestion — bridges pybit WS thread to asyncio
    context_poller.py   # long/short ratio REST poller (only REST poller)
    historical_loader.py # offline loader: quote-saver.bycsi.com → daily SQLite
  features/             # Phase 2 — complete
    replay.py           # load daily DBs into synchronous dicts
    ob_serializer.py    # serialize_ob_snapshot → [K, 4] raw LOB tensor
    flow_features.py    # 9 trade-flow features per 100ms bucket
    context_features.py # ContextBuilder — 17 forward-filled ctx features
    labels.py           # multi-horizon smooth mid-price labels
    normalizer.py       # rolling z-score — flow features only
    pipeline.py         # @hydra.main entrypoint, wires everything
  models/               # Phase 3
    encoders.py         # LOBEncoder (CNN2d), FlowEncoder (GRU/Mamba), CtxEncoder (MLP)
    fusion.py           # GatedFusion
    hybrid.py           # HybridSignalModel — encoders + fusion + head
  training/             # Phase 3
    dataset.py          # LOBDataset — loads ob_raw + flow + ctx, windowed
    losses.py           # MultiHorizonLoss — CE + focal option, class weights
    metrics.py          # per-horizon F1, accuracy, confusion matrix
    train.py            # LightningModule + @hydra.main entrypoint
  inference/            # Phase 5
    buffer.py           # RingBuffer — sliding window for live stream
    preprocessor.py     # online feature extraction + normalization
    predictor.py        # loads model + stats, predict(snapshot) → signal
  signals/              # Phase 4
  risk/                 # Phase 4
  execution/            # Phase 5
  backtest/             # Phase 5
conf/
  config.yaml           # defaults list + hydra (run.dir: ., output_subdir: null)
  ingestion/default.yaml
  features/default.yaml
  model/default.yaml    # Phase 3
  train/default.yaml    # Phase 3
  historical/default.yaml  # historical data loader params
tests/
  data/                 # Phase 1
  features/             # Phase 2
  training/             # Phase 3
```

## Configuration — Hydra + OmegaConf, pure YAML

All tuneable parameters live in `conf/` as plain YAML. No typed config dataclasses / Structured Configs anywhere — access via `cfg: DictConfig`.

Rules:
- Entry points (`bot/main.py`, `bot/features/pipeline.py`, future `bot/training/train.py`) use `@hydra.main(config_path="...", config_name="config", version_base=None)`
- `main(cfg)` extracts values and passes them explicitly to inner functions
- Inner functions never import or reference `cfg` — take plain args with type hints
- CLI override: `python -m bot.features.pipeline features.start_date=2026-04-12`
- Secrets (API keys): env vars via python-dotenv, never in YAML

Config files:
- **conf/config.yaml** — `defaults` list, `hydra.run.dir: .`, `hydra.output_subdir: null` (no cwd change, no `outputs/` clutter)
- **conf/ingestion/default.yaml** — `symbol`, `data_dir`, `lob_depth`, `snapshot_interval_s`, `queue_maxsize`, `ls_ratio_interval_s`
- **conf/features/default.yaml** — `start_date`, `end_date`, `bucket_ms`, `train_ratio`, `val_ratio`, `normalizer_window`, `label_alpha`, `label_horizons`, `oi_history_window_ms`, `liq_window_ms`
- **conf/model/default.yaml** (Phase 3) — `seq_len`, `ob_depth`, `in_features_flow`, `in_features_ctx`, `d_model`, `d_ctx`, `n_lob_blocks`, `flow_encoder` (`gru`|`mamba`), `gru_layers`, `n_classes`, `dropout`, `use_ctx`
- **conf/train/default.yaml** (Phase 3) — `batch_size`, `max_epochs`, `lr`, `weight_decay`, `patience`, `grad_clip_val`, `use_focal_loss`, `focal_gamma`, `val_metric`, `seed`
- **conf/historical/default.yaml** — `start_date`, `end_date`, `base_url`, `snapshot_interval_ms` (downsample target; source is 10 ms)

Note: `cfg.ingestion.lob_depth` is the single source of truth for K (LOB levels per side). `cfg.model.ob_depth` must match it.

## Bybit API

### REST (public, no auth)
- Mainnet: `https://api.bybit.com`, Demo: `https://api-demo.bybit.com`
- Long/short ratio: `GET /v5/market/account-ratio` — `category=linear, symbol=BTCUSDT, period=15min, limit=1`. Polled every 15 min — only remaining REST poller.

### WebSocket public — `wss://stream.bybit.com/v5/public/linear`
Four topics on one connection:
- `orderbook.50.BTCUSDT` — order book
- `publicTrade.BTCUSDT` — trades
- `tickers.BTCUSDT` — mark/index price, OI, funding, volume
- `allLiquidation.BTCUSDT` — real-time liquidation events

### pybit init
- `HTTP(api_key=..., api_secret=..., demo=True)`
- `WebSocket(channel_type="linear", demo=True)`

### pybit threading model
WebSocket runs in a daemon thread; callbacks execute in the WS thread. `OrderBookManager.update()` is called from the WS thread and guarded by `threading.Lock`. All other callbacks bridge to asyncio via `loop.call_soon_threadsafe()`. pybit applies OB and ticker deltas internally and always delivers a full snapshot to the callback.

### OrderBookManager reconnect detection
pybit auto-reconnects and resubscribes. Book resets on reconnect, detected via sequence number regression (`seq < last_seq`). First snapshot after init or reconnect has `is_reset=True`.

### StorageWriter backpressure
Queue full → drop oldest, increment counter, log at next write cycle. Default `queue_maxsize=10_000`.

### Graceful shutdown order in main.py
1. `ingestion.stop()` + `ingestion_task.cancel()` — stop WS feed
2. `await gather(ls_ratio_task)` — poller finishes current poll
3. `await storage_task` — StorageWriter drains queue, closes DB

## Historical data loader (`bot/data/historical_loader.py`)

Offline synchronous script that downloads Bybit public historical data from `https://quote-saver.bycsi.com` and writes it to the same daily SQLite schema as Phase 1 — so Phase 2 runs unchanged on historical data.

Sources:
- Order book: `{base_url}/orderbook/linear/{SYMBOL}/{DATE}_{SYMBOL}_ob500.data.zip` — JSONL, one full snapshot per line at 10 ms cadence, 500 levels each side, fields `ts`, `data.b`, `data.a`, `data.seq`.
- Trades: `{base_url}/trade/linear/{SYMBOL}/{DATE}_{SYMBOL}.csv.gz` — CSV (header auto-detected). Columns: `timestamp, symbol, side, size, price, [tickDirection, trdMatchID, ...]`.

Processing:
- Downsample: keep every `snapshot_interval_ms / 10` snapshot (default 100 ms → every 10th).
- Depth slice: keep only `cfg.ingestion.lob_depth` levels per side.
- `is_reset=True` on first snapshot of each day + whenever `seq < last_seq`.
- Only `snapshots` and `trades` tables — historical source has no ticker/liquidations/LS ratio.
- Resumable: skips a day if `{db_dir}/{SYMBOL}_{DATE}.db` already exists.
- No retry on network failure — rerun the command, completed days are skipped.

DDL for `snapshots`/`trades` is re-exported from `storage.py` (`SNAPSHOTS_DDL`, `TRADES_DDL`, plus matching `*_INDEX` names) so the historical loader and `StorageWriter` share one schema definition.

Run: `python -m bot.data.historical_loader historical.start_date=2025-01-01 historical.end_date=2025-01-07`.

## Data schema (SQLite, all tables in `{SYMBOL}_{YYYY-MM-DD}.db`)

### snapshots
`timestamp_ms INTEGER, is_reset INTEGER, bids TEXT, asks TEXT`
bids/asks: JSON list of `[price_str, qty_str]`, up to 50 levels, best first.

### trades
`timestamp_ms INTEGER, trade_id TEXT, side TEXT, price TEXT, qty TEXT`

### ticker_context
`timestamp_ms, mark_price, index_price, open_interest, open_interest_value, funding_rate, next_funding_time, price_24h_pct, prev_price_1h, volume_24h, turnover_24h, collected_at_ms`
All price/qty fields nullable TEXT (Decimal-safe). NULLs normal for delta messages — forward-filled in Phase 2.

### liquidations
`timestamp_ms, side, qty, price, collected_at_ms`

### long_short_ratio
`timestamp_ms UNIQUE, buy_ratio, sell_ratio, collected_at_ms` — `INSERT OR IGNORE` + in-memory `ts > last_ts` dedup on restart.

## Phase 2 output (Phase 3 input)

Files in `data/features/{SYMBOL}/`:
- `ob_raw.npy`        `[N, K, 4]` float32 — raw LOB tensor, NOT normalized
- `flow_features.npy` `[N, 9]`    float32 — z-score normalized trade flow
- `ctx_features.npy`  `[N, 17]`   float32 — NOT normalized, forward-filled
- `labels.npy`        `[N, H]`    int8    — classes 0/1/2, H horizons
- `flat_mask.npy`     `[N, H]`    bool    — True = flat sample, skip in training
- `timestamps_ms.npy` `[N]`       int64
- `is_reset.npy`      `[N]`       bool
- `normalizer_stats.npz`          — rolling z-score params for flow features only

`N` = valid snapshots (reset zones excluded in labels, not in arrays).
`K` = `cfg.ingestion.lob_depth` (default 50).
`H` = `len(cfg.features.label_horizons)` (default 3: `[1, 5, 10]`).
Classes: `0=down, 1=flat, 2=up`.

### LOB tensor encoding
Raw levels encoded as offsets from best bid/ask — stationary across price regimes:
```
ob_raw[n, k, 0] = (best_bid - bids[k].price) / best_bid    # bid price offset, ≥0
ob_raw[n, k, 1] = log1p(bids[k].qty)                       # bid size
ob_raw[n, k, 2] = (asks[k].price - best_ask) / best_ask    # ask price offset, ≥0
ob_raw[n, k, 3] = log1p(asks[k].qty)                       # ask size
```
Missing levels (fewer than K): padded with zeros. Self-normalizing — no z-score.

### Trade flow features (9 dims, per `bucket_ms` aligned to snapshot ts)
`buy_volume` (log1p), `sell_volume` (log1p), `signed_volume`, `trade_count`, `ofi = signed_volume / (buy_volume + sell_volume + 1e-8)`, `max_trade_size` (log1p), `mean_trade_size` (log1p, 0 if no trades), `buy_count`, `sell_count`. Rolling z-score, `window=normalizer_window`, fit on train split only.

### Context features (17 dims, forward-filled, NOT normalized)
From ticker: `mark_price`, `index_price`, `basis_bps`, `open_interest`, `oi_change_pct_1h`, `funding_rate`, `funding_sign`, `volume_24h`, `price_24h_pct`, `open_interest_value`.
From liquidations (1-min window): `liq_buy_vol_1m`, `liq_sell_vol_1m`, `liq_imbalance_1m`, `liq_count_1m`.
From long/short ratio: `ls_buy_ratio`, `ls_sell_ratio`, `ls_ratio`.

### Labels (smooth mid-price, multi-horizon)
```
prev = mean(mid_price[t-h : t])
fut  = mean(mid_price[t : t+h])
ret  = (fut - prev) / prev
```
`|ret| < alpha` → class 1 (flat, flagged in `flat_mask`). `ret > alpha` → 2, `< -alpha` → 0. Reset zones (`horizon` snapshots before and after each `is_reset`) excluded via `valid_mask`.

### Normalization
- Rolling z-score, incremental, window=`cfg.features.normalizer_window`
- Fit on train split only (`train_ratio`), frozen for val+test
- Saved stats in `normalizer_stats.npz` — reused in Phase 5 online inference
- Scope: flow features only. LOB tensor self-normalizing; ctx not normalized.

## Architecture (Phase 3)

### LOBEncoder
Input `[B, T, K, 4]` → permute `[B, 4, T, K]` → `n_lob_blocks` Conv2d blocks → `AdaptiveAvgPool2d` over K → `[B, T, d_model]`.
Blocks: asymmetric kernels `(1,3)` for first block (spatial only), `(3,3)` for rest. Channels: `4 → 32 → 64 → 128` (3 blocks). Each block: `Conv2d + BatchNorm2d + GELU + Dropout2d`.

### FlowEncoder
Input `[B, T, 9]` → GRU (default, `gru_layers` layers) or Mamba (opt-in via `cfg.model.flow_encoder="mamba"`; ImportError with install hint if `mamba_ssm` missing) → `[B, T, d_model]`.

### CtxEncoder
Input `[B, T, 17]` (actual ctx values at each timestep, not repeated) → per-timestep `Linear(17 → d_ctx) + GELU + LayerNorm` → `[B, T, d_ctx]`. Disabled when `cfg.model.use_ctx=False`.

### GatedFusion
Input: `z_lob [B,T,d_model]`, `z_flow [B,T,d_model]`, `z_ctx [B,T,d_ctx]` (optional). `use_ctx: bool` fixed at init (clean ONNX export). Concat → candidate + gate projections → sigmoid gating → `[B, T, d_model]`. Interface stable for future swap to cross-attention.

### Head
Global mean pool over T → Linear → `[B, H, C]` logits.

### Training
- Loss: `MultiHorizonLoss` — CE (or focal) across all H horizons, masked mean via `flat_mask`.
- `flat_mask` zeroes out flat positions in loss tensor: `(ce * ~flat_mask).sum() / (~flat_mask).sum()`.
- Class weights: inverse-frequency, computed from train split labels.
- Train on all H horizons jointly (shared encoders benefit from multi-task gradient).
- Val: per-horizon F1 macro + accuracy, checkpoint on `cfg.train.val_metric` (default `val_f1_macro`).
- `n_horizons` derived from `len(cfg.features.label_horizons)` — not in model config.

## Online vs offline data flow

Same model both paths; only data prep differs.

Offline (training):
- `ob_raw.npy` → `LOBDataset` window → `LOBEncoder`
- `flow_features.npy` (pre-normalized) → window → `FlowEncoder`
- `ctx_features.npy` (pre-filled) → window → `CtxEncoder`

Online (Phase 5):
- WS snapshot → `ob_serializer` → `RingBuffer[T, K, 4]` → `LOBEncoder`
- WS trades → `TradeAccumulator` → flow extractor → `Normalizer.normalize_only()` → `RingBuffer[T, 9]`
- WS ticker/liquidations/ls_ratio → `ContextCache` → `RingBuffer[T, 17]`

`normalizer_stats.npz` used identically offline and online — flow features only.

## Coding conventions
- All tuneable numbers via `cfg` — never hardcoded
- Prices/quantities: `Decimal` in memory, `str` in DB, `float32` at the feature boundary
- Timestamps: UTC, unix ms (int)
- Feature pipeline synchronous — offline, not in event loop
- Every public function has type hints
- Logging via `structlog` — only at pipeline entry points (`main.py`, `pipeline.py`, `train.py`), not inside extractors/encoders
- Tests: synthetic data (in-memory sqlite OK for pipeline tests, no network / no live files)

## Code style — strict rules
- No proxy/wrapper classes around the SDK
- No `try/except` unless a specific known exception with real recovery
- No logging inside business-logic functions
- No defensive "just in case" code
- No comments that restate what the code does — explain *why* if non-obvious

## Interaction style
- Before implementing, ask clarifying questions if requirements are ambiguous
- When multiple approaches exist, present trade-offs and ask which to take
- Don't assume on decisions with long-term consequences — flag and ask
- After completing a task, briefly summarize what was done and what comes next

## Known decisions
- Bybit Demo (not testnet) — real market data, virtual funds
- Single-symbol MVP (BTCUSDT); multi-symbol later
- SQLite daily rotation; all tables in one daily file
- Snapshot interval 100ms, LOB depth 50
- Trade flow bucket 100ms, aligned to snapshot timestamps
- Ticker WS replaces REST OI + funding polling (better granularity, fewer connections)
- All four public WS topics share one pybit WebSocket connection
- Long/short ratio: only remaining REST poller, 15-min interval
- Ticker delta NULLs: forward-filled in Phase 2, not at ingestion
- LOB: raw tensor `[N, K, 4]` with price offsets + log1p sizes — CNN learns patterns
- LOB self-normalizing (offsets + log1p) — no z-score for LOB
- Flow: rolling z-score, window=500, fit on train only
- Context: not normalized, forward-filled
- Label method: smooth mid-price, `alpha=0.001`, horizons `[1, 5, 10]`
- Flat samples kept as class 1 + `flat_mask` — not dropped at extraction
- Train/val/test: chronological 80/10/10
- `flat_mask`: NOT applied in dataset — applied in the training step before loss
- Walk-forward split only; boundaries aligned with normalizer stats
- Checkpoint on best `val_f1_macro`, early stopping via `cfg.train.patience`
- AMP enabled if CUDA available
- GRU default flow encoder, Mamba opt-in via config
- `ctx` toggleable via `cfg.model.use_ctx` for ablation
- LOBEncoder: 2D conv with asymmetric kernels `(1,3)→(3,3)→(3,3)`, AdaptiveAvgPool2d over K
- ctx in Dataset: actual values at each timestep (option a), not repeated — CtxEncoder is per-timestep MLP, no overfitting risk
- GatedFusion `use_ctx` fixed at init (not dynamic) — clean ONNX export, different checkpoints for different architectures
- Multi-horizon training: joint loss across all H, not single-horizon — better gradient signal for shared encoders
- Class weights: inverse-frequency from train labels, applied inside CE
- `n_horizons` computed from `cfg.features.label_horizons`, not stored in model config
- Config pattern: Hydra + OmegaConf, pure YAML — no Structured Configs / typed dataclasses
- Historical backfill: `bot/data/historical_loader.py` synchronously downloads Bybit public archive (quote-saver.bycsi.com), produces the same daily SQLite files Phase 1 writes — so Phase 2 is source-agnostic
- Historical data has only `snapshots` + `trades` — ticker/liquidations/LS ratio are live-only (Phase 1 ingestion)
- `requests` dependency added for historical loader; `SNAPSHOTS_DDL`/`TRADES_DDL`/`*_INDEX` exported from `storage.py` for reuse
