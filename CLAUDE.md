# Trading Bot — Bybit Perpetual Futures

## Project goal
ML-driven trading bot for Bybit USDT-margined perpetual futures.
Phase 1 (ingestion), Phase 2 (offline feature pipeline), Phase 3 (model + training),
Phase 4 (signals/risk) and Phase 5 (inference/execution/backtest) complete.
Next: train on real data, then run the backtest and a Demo dry run.

## Stack
- **Exchange:** Bybit v5 API, pybit v5 — used directly, no proxy classes
- **Async:** asyncio + uvloop
- **Storage:** SQLite daily rotation (`{SYMBOL}_{YYYY-MM-DD}.db`), aiosqlite for async writes
- **Config:** Hydra + OmegaConf — pure YAML under `conf/`, no typed dataclasses / Structured Configs
- **Features:** numpy .npy arrays (Phase 2 output)
- **ML:** PyTorch + PyTorch Lightning, CUDA (local GPU) — Phase 3+
- **Backtest:** hftbacktest (Rust core) — no in-house engine; our data is adapted to it
- **Secrets:** environment variables only (python-dotenv), never in YAML

## Project structure
```
bot/
  main.py               # Phase 1 entrypoint, @hydra.main
  data/                 # Phase 1 — complete
    models.py           # dataclasses: OrderBookSnapshot, Trade, TickerContext, Liquidation, LongShortRatio
    orderbook.py        # thread-safe container for pybit WS book state
    storage.py           # async queue consumer, SQLite daily rotation
    ws_client.py         # DataIngestion — bridges pybit WS thread to asyncio
    context_poller.py    # long/short ratio REST poller (only REST poller)
    historical_loader.py # offline loader: quote-saver.bycsi.com → raw cache + daily SQLite
    observer.py         # MarketObserver protocol — optional live tap on the stream
  features/             # Phase 2 — complete
    replay.py            # load daily DBs into synchronous dicts
    ob_serializer.py      # serialize_ob_snapshot → [K, 4] raw LOB tensor
    flow_features.py      # 9 trade-flow features per 100ms bucket
    context_features.py   # ContextBuilder — 17 forward-filled ctx features
    labels.py             # multi-horizon smooth mid-price labels
    normalizer.py         # rolling z-score — flow features only
    pipeline.py           # @hydra.main entrypoint, wires everything
  models/                # Phase 3
    encoders.py           # LOBEncoder (CNN2d), FlowEncoder (GRU/Mamba), CtxEncoder (MLP)
    fusion.py              # GatedFusion
    hybrid.py              # HybridSignalModel — encoders + fusion + head
  training/              # Phase 3
    dataset.py             # LOBDataset — loads ob_raw + flow + ctx, windowed
    losses.py              # MultiHorizonLoss — CE + focal option, class weights
    metrics.py             # per-horizon F1, accuracy, confusion matrix
    train.py                # LightningModule + @hydra.main entrypoint
  inference/             # Phase 5 — complete
    buffer.py               # RingBuffer — sliding window for live stream
    accumulator.py          # TradeAccumulator — trades between two snapshots
    adapters.py             # WS dataclasses → Phase 2 extractor row shapes
    preprocessor.py         # online feature extraction + normalization
    predictor.py            # loads model + stats, predict(...) → InferenceResult
  signals/               # Phase 4 — complete
    types.py                # SignalAction, SignalPolicy, TradingSignal
    policy.py               # generate_signal — argmax + confidence gate
  risk/                  # Phase 4 — complete
    types.py                # MarketState, PositionState, RiskPolicy, RiskDecision
    gate.py                 # evaluate_risk — spread/stale/cooldown/notional/leverage
  execution/             # Phase 5 — complete
    types.py                # OrderSide/Type, ExitPolicy/Reason, OpenPosition, OrderIntent
    lifecycle.py            # pure: entry sizing, TP/SL/hold exit, realized PnL
    orders.py               # OrderIntent → pybit place_order/cancel_order kwargs
    account.py              # AccountState — position/wallet/order from private WS
    live.py                 # LiveTrader — MarketObserver + order state machine
    run.py                  # @hydra.main live entrypoint
  telemetry/             # cross-cutting — what the live loop reports about itself
    types.py                # PipelineCounters (shared with backtest), DecisionRecord, OrderEvent
    collector.py            # counters, quantile gauges, flow-feature skew, window rollups
    evaluation.py           # predicted vs realized, scored with the training label rule
  backtest/              # Phase 5 — complete, built on hftbacktest
    prepare.py              # cached bycsi archives → hftbacktest feed/snapshot/latency
    strategy.py             # the loop hftbacktest drives; reuses signals/risk/lifecycle
    report.py               # self-contained HTML report; also a CLI to redraw a saved run
    run.py                  # @hydra.main entrypoint — scores a split, replays it
conf/
  config.yaml            # defaults list + hydra (run.dir: ., output_subdir: null)
  ingestion/default.yaml
  features/default.yaml
  model/default.yaml     # Phase 3
  train/default.yaml     # Phase 3
  historical/default.yaml
  signals/default.yaml   # Phase 4
  risk/default.yaml      # Phase 4
  inference/default.yaml # Phase 5
  execution/default.yaml # Phase 5 — order + exit policy, shared with backtest
  backtest/default.yaml  # Phase 5 — split, fees, queue/latency models, feed paths
  telemetry/default.yaml # rollup interval, quantile window size
tests/
  data/                  # Phase 1
  features/              # Phase 2
  training/              # Phase 3
  signals/ risk/         # Phase 4
  inference/ execution/ backtest/   # Phase 5
  telemetry/             # counters, skew, live scoring, storage round-trip
```

## Configuration — Hydra + OmegaConf, pure YAML

All tuneable parameters live in `conf/` as plain YAML. No typed config dataclasses / Structured Configs anywhere — access via `cfg: DictConfig`.

Rules:
- Entry points use `@hydra.main(config_path="...", config_name="config", version_base=None)`
- `main(cfg)` extracts values and passes them explicitly to inner functions
- Inner functions never import or reference `cfg` — take plain args with type hints
- CLI override: `python -m bot.features.pipeline features.start_date=2026-04-12`
- Secrets (API keys): env vars via python-dotenv, never in YAML

Config files (key fields):
- **conf/config.yaml** — `defaults` list, `hydra.run.dir: .`, `hydra.output_subdir: null`
- **conf/ingestion/default.yaml** — `symbol`, `data_dir`, `lob_depth`, `snapshot_interval_s`, `queue_maxsize`, `ls_ratio_interval_s`
- **conf/features/default.yaml** — `start_date`, `end_date`, `bucket_ms`, `train_ratio`, `val_ratio`, `normalizer_window`, `label_alpha`, `label_horizons`, `oi_history_window_ms`, `liq_window_ms`
- **conf/model/default.yaml** — `seq_len`, `ob_depth`, `in_features_flow`, `in_features_ctx`, `d_model`, `d_ctx`, `n_lob_blocks`, `flow_encoder` (`gru`|`mamba`), `gru_layers`, `n_classes`, `dropout`, `use_ctx`
- **conf/train/default.yaml** — `batch_size`, `max_epochs`, `lr`, `weight_decay`, `patience`, `grad_clip_val`, `use_focal_loss`, `focal_gamma`, `val_metric`, `seed`
- **conf/historical/default.yaml** — `start_date`, `end_date`, `base_url`, `snapshot_interval_ms`
- **conf/signals/default.yaml** — `horizon_index`, `min_confidence`, `allow_short`
- **conf/risk/default.yaml** — `max_position_notional`, `max_leverage`, `max_spread_bps`, `stale_data_ms`, `cooldown_after_trade_ms`, `min_confidence`, `allow_short`
- **conf/inference/default.yaml** — `checkpoint_path`, `normalizer_stats_path`, `device`, `warmup_action`
- **conf/execution/default.yaml** — `order_notional`, `entry_order_type`, `exit_order_type`, `order_timeout_ms`, `exit_fallback_ms`, `exit_repeg`, `cross_on_stop`, `hold_snapshots`, `take_profit_bps`, `stop_loss_bps`, `qty_step`, `price_tick`, `min_order_qty`, `demo`, `dry_run`
- **conf/backtest/default.yaml** — `split`, `checkpoint_path`, `initial_equity`, `fill_model`, `maker_fee_bps`, `taker_fee_bps`, `device`, `batch_size`, `tag` (run-directory suffix, so a sweep keeps every run), `prediction_cache_dir` (model scores keyed by checkpoint+split — a config sweep never varies anything the model sees)

Note: `cfg.ingestion.lob_depth` is the single source of truth for K (LOB levels per side). `cfg.model.ob_depth` must match it.

## Bybit API — quick reference

- REST (public): `https://api.bybit.com` (mainnet) / `https://api-demo.bybit.com` (demo). Long/short ratio via `GET /v5/market/account-ratio` — only remaining REST poller, 15-min interval.
- WS public: `wss://stream.bybit.com/v5/public/linear` — four topics on one connection: `orderbook.50.BTCUSDT`, `publicTrade.BTCUSDT`, `tickers.BTCUSDT`, `allLiquidation.BTCUSDT`.
- pybit init: `HTTP(api_key=..., api_secret=..., demo=True)`, `WebSocket(channel_type="linear", demo=True)`.
- Threading: WS runs in daemon thread; `OrderBookManager.update()` guarded by `threading.Lock`; other callbacks bridge via `loop.call_soon_threadsafe()`. pybit always delivers full snapshots.
- Reconnect: detected via sequence regression (`seq < last_seq`); first snapshot after init/reconnect has `is_reset=True`.
- StorageWriter backpressure: queue full → drop oldest, log at next cycle. Default `queue_maxsize=10_000`.
- Shutdown order: `ingestion.stop()` → drain `ls_ratio_task` → drain `storage_task`.

> Full historical-loader mechanics and SQLite table schemas: see `docs/reference.md`.

## Phase 2 output (Phase 3 input)

Files in `data/features/{SYMBOL}/`: `ob_raw.npy [N,K,4]`, `top_of_book.npy [N,2]` (best_bid/best_ask float64), `flow_features.npy [N,9]` (z-scored), `ctx_features.npy [N,17]` (raw, forward-filled), `labels.npy [N,H]` (0/1/2), `flat_mask.npy [N,H]`, `valid_mask.npy [N]`, `timestamps_ms.npy [N]`, `is_reset.npy [N]`, `normalizer_stats.npz`.
`top_of_book.npy` holds absolute prices, which `ob_raw` cannot express — it stores only offsets from the best price. The hftbacktest path takes prices from the replayed book instead, so this file is now diagnostic: it is what lets you check the feature grid against the feed.
`K = cfg.ingestion.lob_depth` (50). `H = len(cfg.features.label_horizons)` (default 3: `[1,5,10]`). Classes: `0=down, 1=flat, 2=up`.

> Full feature encodings (LOB tensor formula, 9 flow features, 17 ctx features, label formula): see `docs/reference.md`.

## Architecture (Phase 3)

- **LOBEncoder**: `[B,T,K,4]` → permute → Conv2d blocks (`4→32→64→128`, kernels `(1,3)` then `(3,3)`, each `Conv2d+BN+GELU+Dropout2d`) → `AdaptiveAvgPool2d` over K → `[B,T,d_model]`.
- **FlowEncoder**: `[B,T,9]` → GRU (default) or Mamba (`cfg.model.flow_encoder="mamba"`, ImportError w/ install hint if missing) → `[B,T,d_model]`.
- **CtxEncoder**: `[B,T,17]` (actual per-timestep values) → `Linear(17→d_ctx)+GELU+LayerNorm` → `[B,T,d_ctx]`. Disabled when `cfg.model.use_ctx=False`.
- **GatedFusion**: concat `z_lob, z_flow, z_ctx` → candidate+gate projections → sigmoid gate → `[B,T,d_model]`. `use_ctx` fixed at init (clean ONNX export).
- **Head**: mean-pool over T → Linear → `[B,H,C]` logits.
- **Training**: `MultiHorizonLoss` (CE or focal), masked mean via `valid_mask` — the rows that carry a usable label. Flat is **in** the loss: it is one of the three classes, and excluding it (which `flat_mask` used to do) left the model no gradient towards predicting flat, so it could only ever pick a direction. Inverse-frequency class weights from the train split over all three classes, all H horizons trained jointly. Val: per-horizon F1 macro + accuracy, checkpoint on `cfg.train.val_metric` (default `val_f1_macro`). `n_horizons` derived from `len(cfg.features.label_horizons)`, not stored in model config.

## Online vs offline data flow

Same model both paths; only data prep differs.
- Offline: `.npy` files → windowed `LOBDataset` → encoders.
- Online (Phase 5): WS snapshot → `ob_serializer` → `RingBuffer` → `LOBEncoder`; WS trades → `TradeAccumulator` → `Normalizer.normalize_only()` → `RingBuffer`; WS ticker/liq/ls_ratio → `ContextBuilder` (inside `OnlinePreprocessor`) → `RingBuffer`.
`normalizer_stats.npz` used identically offline and online — flow features only.
`bot/inference/adapters.py` converts the live WS dataclasses into the row shapes the Phase 2 extractors already take, so both paths run the exact same extraction code.

## Signals, risk and execution (Phases 4–5)

- **Signal** (`bot/signals/policy.py`): argmax over `class_probs[horizon_index]`. Flat → `NO_TRADE`; below `min_confidence` → `NO_TRADE`; down with `allow_short=False` → `NO_TRADE`. Every outcome carries a `reason`.
- **Risk gate** (`bot/risk/gate.py`): rejects in order — not actionable, symbol mismatch, stale market data, spread too wide, low confidence, shorting disabled, cooldown, resulting notional, leverage.
- **Position lifecycle** (`bot/execution/lifecycle.py`): pure functions shared by the backtest and the live loop, so both agree by construction. Entry qty is sized off the price the order actually trades at — passive side for post-only, crossing side for market — so the notional the risk gate approved is the notional taken on.
- **Exit**: fixed hold (`hold_snapshots × bucket_ms`, defaulting to the trained horizon) with protective TP/SL in bps. Priority when a snapshot straddles several: stop-loss → take-profit → hold expiry. TP/SL are trigger levels, not order prices, so they stay off the tick grid.
- **Exit escalation**: the exit is posted passively and **re-pegged** to the top of its own side whenever the book leaves it behind (`should_repeg_exit`); it is crossed only after `cfg.execution.exit_fallback_ms`. A market exit costs 5.5 bps taker against 2 bps maker (VIP0 — this account is not on the market-maker programme, so maker is a smaller cost, not a rebate) — more than the whole expected move at these horizons — so crossing is the bounded last resort, not the reflex. A stop breach alone no longer forces a cross (`cfg.execution.cross_on_stop=false`): paying 5.5 bps to cap a 5 bps stop costs more than the loss it prevents. But re-pegging cannot be unconditional either — in a one-way market the ask is gone before anyone lifts it, so the chase never completes and the position never leaves; the fallback is what turns "cheaper per fill" back into "bounded".
- **Order types**: post-only limit entry resting on the passive side (2 bps maker vs 5.5 bps taker at VIP0 — cheaper, not a rebate), market exit (protective exits must fill). Both configurable via `cfg.execution`. Post-only is a Limit order with `timeInForce=PostOnly`, not its own order type.
- **Live state** (`bot/execution/account.py`): position, wallet equity and own orders come from the Bybit private WS, matched to our intents by `orderLinkId`. Guarded by `threading.Lock` like `OrderBookManager` — pybit delivers on its own thread.
- **LiveTrader** (`bot/execution/live.py`): implements `MarketObserver`, so it consumes the same stream `StorageWriter` records. Snapshots are queued (drop-oldest) rather than acted on inline — the decision step awaits blocking pybit REST calls and must not stall ingestion. One position at a time; a market entry still waits for the order stream to report the real average fill price before the exit bracket is built.
- **Backtest** (`bot/backtest/`): hftbacktest replays the untouched 10 ms / 500-level Bybit feed; we only supply the strategy. `prepare.py` converts the archives `historical_loader` caches, `strategy.py` walks the feed one bucket at a time calling the same pure functions as `LiveTrader`, and metrics come from `hftbacktest.stats`. Deliberately plain Python, not a numba `@njit` kernel — that is what lets backtest and live share code.
- **Schedule alignment**: the model runs on the 100 ms feature grid while the feed is tick-level, so predictions are precomputed and attached to their snapshot timestamps. Row `j` scores snapshot `seq_len - 1 + j`. The feed covers whole days but a split does not: past the last prediction the strategy stops entering and leaves once flat, so it never trades on a stale signal.

- **Reporting**: every run writes `record.npz`, `run.json` and `report.html` into `{report_dir}/{SYMBOL}_{split}/`, and `python -m bot.backtest.report <run_dir>` redraws the page from those without replaying. Equity, pre-fee equity and buy-and-hold are all indexed to percent of book size so they share one axis — hftbacktest's own plot puts price on a twin axis, which cannot be read by eye. The decision funnel (actionable → approved → submitted → filled / cancelled / expired) is ours alone; it is what says whether a weak result is the model's fault or the execution's.

> Data preparation, queue/latency models, the latency caveat and the report: see `docs/reference.md`.

## Telemetry — attributing failure to a stage

Every gauge exists to answer *which stage broke*, so a bad session points at a knob
rather than at "the bot lost money".

- **One funnel type for live and backtest** (`PipelineCounters`). Run a backtest over the
  window a live session covered and subtract: the stage where they diverge is the
  modelling error. The signal policy's three NO_TRADE reasons and the risk gate's nine
  rejection reasons are counted by name, so a rejection is never just a missing count.
- **Every loop pass is recorded**, not only the ones that trade. `NO_BOOK`, `NOT_READY`,
  `HOLDING` and `WAITING` say the loop could not act at all, which is a data or warm-up
  problem rather than a model problem, and is invisible if only fills are logged.
- **Train/serve skew** is the first thing to check. `OnlinePreprocessor` z-scores against
  the mean and std frozen from the train split, so a live stream matching training gives
  `z ~ N(0,1)`; `FlowSkew` reports the worst feature by drift of mean (level shift) and of
  std (variance blow-up). No execution tuning fixes a model being asked about a
  distribution it never saw.
- **Context staleness is tracked separately** because `ContextBuilder` forward-fills
  indefinitely: a dead ticker stream is indistinguishable from a quiet one from the
  features alone, and 17 ctx features silently freeze.
- **Order latency is measured**, submit → fill and submit → ack. This is exactly the value
  `cfg.backtest.feed_latency_ns` currently assumes; a Demo session of these is what turns
  `intp_order_latency` from a guess into data.
- **Predicted vs realized** (`telemetry/evaluation.py`) grades live calls with
  `compute_labels` — the same smooth mid-price rule the training set was labelled with — so
  the result is directly comparable to the validation F1 the checkpoint was chosen on.
  Hit rate is also broken down by confidence band: if it does not rise with confidence,
  `min_confidence` is filtering on noise.

Storage: `decisions` and `order_events` land in the same daily SQLite as the market data,
so scoring is a join. Aggregates go to structlog every `cfg.telemetry.rollup_interval_s`.

> Table schemas, the full metric list and how each maps to a knob: see `docs/reference.md`.

## Coding conventions
- All tuneable numbers via `cfg` — never hardcoded
- Prices/quantities: `Decimal` in memory, `str` in DB, `float32` at the feature boundary
- Timestamps: UTC, unix ms (int)
- Feature pipeline synchronous — offline, not in event loop
- Every public function has type hints
- Logging via `structlog` — only at pipeline entry points (`main.py`, `pipeline.py`, `train.py`, `backtest/run.py`, `execution/run.py`) and in `LiveTrader`'s order lifecycle, never inside extractors/encoders or the pure signal/risk/lifecycle functions
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

## Open / less obvious decisions worth remembering
- Bybit Demo (not testnet) — real market data, virtual funds
- Single-symbol MVP (BTCUSDT); multi-symbol later
- Ticker WS replaces REST OI + funding polling
- Label method: smooth mid-price; flat samples kept as class 1 + `flat_mask`, not dropped. `alpha` and the horizons are **measured, not assumed**: on SOLUSDT `alpha=0.001` with sub-second horizons labels 100% of samples flat, because the mean 1-second move is 0.5 bps. The defaults (`alpha=0.0001`, horizons `[50,100,300]` = 5/10/30 s) come from a grid over real data and give ~31/38/32 at the middle horizon
- Train/val/test: chronological 80/10/10, walk-forward split, boundaries aligned with normalizer stats
- `flat_mask` means "this label is flat", a class to be learned; `valid_mask` means "this row has a usable label" and is the one the loss and the metrics exclude on. Conflating them is what killed the flat class
- AMP enabled if CUDA available
- GRU default flow encoder, Mamba opt-in
- `ctx` toggleable via `cfg.model.use_ctx` for ablation
- Exit = fixed hold + protective TP/SL (not TP/SL alone, and not hold-until-opposite-signal) — the hold keeps the backtest directly comparable to the horizon the model was trained on
- Execution is measured on **val**, never on test: the test split has been read in detail, so tuning against it would be fitting the answer. The winning setting moved the val split from −1.76 to −0.04 bps a trade, entirely through the exit
- A checkpoint records the architecture it was trained with (`hyper_parameters.arch`) and that wins over `cfg.model` when loading. Before this, `cfg.model.use_ctx` defaulted to a value the shipped checkpoint was not trained with, and the mismatch surfaced as a tensor-shape stack trace
- Post-only limit entries because maker (2 bps) undercuts taker (5.5 bps) at this account's VIP0 tier — there is no rebate; unfilled orders are cancelled after `order_timeout_ms` rather than chased
- **Fee tier is VIP0, confirmed** — maker +0.02%, taker +0.055%, both costs. Every earlier note in this project calling maker a "rebate" assumed the market-maker programme without checking the account
- Position and equity come from the private WS, not REST polling — `ls_ratio` stays the only REST poller
- Backtesting is hftbacktest's job, not ours: the in-house engine was deleted rather than kept alongside, so there is one fill model and one set of numbers
- The backtest feeds on the raw bycsi archives, not our SQLite — our stored slice is 100 ms / 50 levels, which would throw away exactly the microstructure hftbacktest exists to model. `historical_loader` therefore caches the archives it used to discard
- Strategy stays pure Python (no `@njit`) so backtest and live run the same `signals` / `risk` / `execution.lifecycle` code
- `cfg.execution.dry_run` decides normally but places nothing — use it for the first Demo run
- Telemetry writes a row per decision (~864k/day, ~130 MB) rather than sampling: the `snapshots` table already writes an order of magnitude more, and sampling would break the predicted-vs-realized join