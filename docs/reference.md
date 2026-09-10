# Reference — schema & data-encoding details

Split out from `CLAUDE.md` because Phase 1–2 are complete and stable; consult only when touching ingestion, the historical loader, or the feature pipeline internals.

## Historical data loader (`bot/data/historical_loader.py`)

Offline synchronous script that downloads Bybit public historical data from `https://quote-saver.bycsi.com` and writes it to the same daily SQLite schema as Phase 1 — so Phase 2 runs unchanged on historical data.

Sources:
- Order book: `{base_url}/orderbook/linear/{SYMBOL}/{DATE}_{SYMBOL}_ob500.data.zip` — JSONL at a
  **100 ms** cadence. Only the first line is `type: snapshot` with 500 levels a side; every later
  line is `type: delta` carrying just the changed levels, with quantity `0` meaning the level is
  gone. The parser carries book state exactly as `OrderBookManager` does live. Fields: `ts`,
  `type`, `data.b`, `data.a`, `data.seq`.
- Trades: `https://public.bybit.com/trading/{SYMBOL}/{SYMBOL}{DATE}.csv.gz` — a **different host**
  from the book archives. CSV, header auto-detected. Columns: `timestamp, symbol, side, size,
  price, tickDirection, trdMatchID, ...`. The timestamp is **float seconds**
  (`1755648000.1385`), not integer milliseconds.

Processing:
- Downsample: keep every `snapshot_interval_ms / 100` message (the source is already 100 ms, so the default keeps every one).
- Depth slice: keep only `cfg.ingestion.lob_depth` levels per side.
- `is_reset=True` on first snapshot of each day + whenever `seq < last_seq`.
- Only `snapshots` and `trades` tables — the archives carry no ticker, liquidation or ratio feed.
  The Phase 2 loaders skip a table that is absent rather than failing, so a historical range simply
  produces constant (zero) context features; train such a range with `model.use_ctx=false`.
- Resumable: skips a day if `{db_dir}/{SYMBOL}_{DATE}.db` already exists.
- No retry on network failure — rerun the command, completed days are skipped.
- The downloaded archives are cached under `cfg.historical.raw_dir` and reused: the
  feature pipeline reads the downsampled SQLite, the backtest feeds the untouched
  archives to hftbacktest. Downloads land on a `.part` name and are renamed on
  completion, so an interrupted transfer is never mistaken for a cache hit.

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

### decisions (live only)
`timestamp_ms, status, action, horizon_index, p_down, p_flat, p_up, predicted_class,
confidence, best_bid, best_ask, book_age_ms, is_reset, inference_us, signal_reason,
risk_reason, flow_z_max, flow_z_abs_mean, ctx_ticker_age_ms, ctx_liq_age_ms,
ctx_ls_age_ms, class_changed`

One row per pass of the decision loop, including passes that took no prediction — `status`
is `NO_BOOK`, `NOT_READY`, `HOLDING`, `WAITING`, `BLOCKED`, `REJECTED` or `APPROVED`, and
only the last three carry meaningful probabilities. Context ages are `-1` when a source has
never been seen.

### order_events (live only)
`timestamp_ms, order_link_id, event, side, order_type, reduce_only, qty, limit_price,
fill_price, decision_price, latency_ms, exit_reason`

`event` is `submitted` / `filled` / `cancelled` / `expired`. `latency_ms` is measured from
submission, and `decision_price` is the price the decision was taken at, so slippage is
`fill − decision`, signed so positive is always a cost.

## Feature encodings (Phase 2)

### LOB tensor encoding
Raw levels encoded as offsets from best bid/ask — stationary across price regimes:
```
ob_raw[n, k, 0] = (best_bid - bids[k].price) / best_bid    # bid price offset, ≥0
ob_raw[n, k, 1] = log1p(bids[k].qty)                        # bid size
ob_raw[n, k, 2] = (asks[k].price - best_ask) / best_ask     # ask price offset, ≥0
ob_raw[n, k, 3] = log1p(asks[k].qty)                        # ask size
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

### Streaming and memory
`run_pipeline` processes one day at a time and writes `ob_raw`, `flow_features`, `ctx_features`
and `top_of_book` straight into memory-mapped `.npy` files. Materialising a single day's snapshots
as Python objects costs ~17.7 GB (≈20.5 KB per snapshot once its levels are parsed), and the LOB
tensor alone is ~10 GB for a two-week range — neither fits in ordinary RAM. Context state carries
across day boundaries, so the days cannot be processed independently. Training opens the same files
with `mmap_mode="r"`.

`compute_labels` takes its window means from a prefix sum. The scalar form is O(N·h) in Python,
which for a million snapshots and horizons in the hundreds dominates the entire pipeline.

### Normalization
- Rolling z-score, incremental, window=`cfg.features.normalizer_window`
- Fit on train split only (`train_ratio`), frozen for val+test
- Saved stats in `normalizer_stats.npz` — reused in Phase 5 online inference
- Scope: flow features only. LOB tensor self-normalizing; ctx not normalized.
## Backtest — hftbacktest (Phase 5)

There is no in-house engine. `hftbacktest` replays the real feed and models the
queue, latency and fills; we supply only the strategy.

Run: `python -m bot.backtest.run backtest.checkpoint_path=lightning_logs/version_0/checkpoints/best.ckpt`

### Data path
`historical_loader` caches the two archives it downloads under `cfg.historical.raw_dir`
(`{DATE}_{SYMBOL}_ob500.data.zip`, `{DATE}_{SYMBOL}.csv.gz`) — the exact pair
`hftbacktest.data.utils.bybithistmktdata.convert` consumes. `bot/backtest/prepare.py`
converts each day into `{feed_dir}/{SYMBOL}_{DATE}.npz`, and also builds:

- `{SYMBOL}_{DATE}_eod.npz` — end-of-day book snapshot, used to seed the following day.
  The day before the range is only used when its archives are already cached; pulling an
  extra day unasked would be a surprise download.
- `{SYMBOL}_{DATE}_latency.npz` — order latency for the `intp` model.

All three are skipped when the output already exists, so re-runs are cheap.

The backtest therefore sees the untouched 10 ms / 500-level feed, **not** the 100 ms /
50-level slice in SQLite. The feature pipeline still reads SQLite; the two paths share
only the timestamps.

### Latency caveat — read before trusting fill rates
Bybit's public history carries no local timestamp, so there is no measured feed latency.
`convert` synthesises one by adding `cfg.backtest.feed_latency_ns` to every exchange
timestamp, and `generate_order_latency` derives order latency from it as
`mul * feed_latency + offset`. The plumbing is real and the models are wired, but the
latency itself is an assumption. Collect a real latency file on Demo before reading
anything into the fill rate.

### Models (`conf/backtest/default.yaml`)
- `queue_model`: `risk_adverse` | `log_prob` | `power_prob` | `power_prob2` | `power_prob3`
  (default `power_prob3` with `queue_power`). `risk_adverse` is the pessimistic choice.
- `latency_model`: `intp` (interpolated from the generated latency files, default) or
  `constant` (`entry_latency_ns` / `response_latency_ns`).
- `no_partial_fill_exchange`, `linear_asset(1.0)`, `trading_value_fee_model(maker, taker)`
  with `maker_fee_rate` negative for the Bybit rebate.
- `initial_equity` exists because hftbacktest has no starting balance:
  `state_values.balance` is the cash flow generated by trading and opens at zero, so the
  risk gate would otherwise divide by an empty account.

### Order semantics
`GTX` is hftbacktest's post-only: a GTX limit that would cross expires unfilled
(status `EXPIRED`), which is exactly the Bybit behaviour the live path relies on. Exits
are `GTC` `MARKET`, so protective exits always fill and pay taker.

Top of book is rebuilt from tick counts, never from `depth.best_bid` / `best_ask`
directly — those are floats and come back as e.g. `100.10000000000001`. A missing side
reads as `NaN`, which is how an incomplete book is detected.

### Schedule alignment
The model scores the 100 ms feature grid; the feed is tick-level. Probabilities are
precomputed for the split and attached to the timestamps they were produced at
(`build_schedule`: row `j` belongs to snapshot `seq_len - 1 + j`). The loop elapses one
`bucket_ms` at a time and takes the newest prediction already published — never one from
the future.

The feed covers whole days but a split usually starts and ends mid-day. Before the first
prediction the loop takes no signal; past the last one it stops opening positions and
breaks once flat, so a stale signal can never be traded.

### Metrics
From `hftbacktest.stats` — `LinearAssetRecord(record).stats(book_size=initial_equity).summary()`.
Two traps: `summary()` **returns** a polars DataFrame rather than printing one, and without
`book_size` the metrics stay in absolute currency instead of returns. Printing a polars
table on a Windows console needs `PYTHONIOENCODING=utf-8` or it dies on `charmap`.

The strategy additionally reports its own counters (signals actionable/approved, orders
submitted/filled/cancelled/expired, exits by reason), which describe decisions rather
than returns.

### Report (`bot/backtest/report.py`)
Each run writes `{report_dir}/{SYMBOL}_{split}/` containing `record.npz` (the raw
recorder array), `run.json` (metrics + strategy counters + run metadata) and
`report.html`. Redraw without replaying: `python -m bot.backtest.report <run_dir>` —
rendering takes milliseconds against minutes for a replay, so the page is regenerated on
every run and the CLI exists for re-rendering old runs after the report itself changes.

`report.html` is one self-contained file: no CDN except the Google Fonts stylesheet, all
data inlined as JSON, charts drawn as SVG in the page. `render_html(data,
standalone=False)` returns the same page without the document shell, for hosts that
supply their own.

Charts and why they are shaped that way:
- **Cumulative return** — net equity, pre-fee equity and buy-and-hold, all as percent of
  book size on a single axis. hftbacktest's built-in plot puts price on a twin axis;
  two scales on one chart cannot be compared by eye, and indexing to a common base is the
  fix. Line ends are direct-labelled, which is also the relief the palette's light-mode
  contrast warning requires.
- **Drawdown** — depth below the running peak. Thinning keeps each bucket's maximum, so
  spikes survive; the other series are stride-sampled.
- **Position**, **decision funnel**, **exit reasons** — the funnel is the diagnostic
  hftbacktest cannot produce.

Series colours are the validated categorical slots 1–3; both themes are defined at token
level, and a table view of the hero chart is available for the contrast relief rule.

## Live execution (Phase 5)

Run: `python -m bot.execution.run` — needs `BYBIT_API_KEY` and `BYBIT_API_SECRET` in the
environment (`.env`, loaded by python-dotenv). Start with `execution.dry_run=true`.

Wiring: one public WS via the existing `DataIngestion` (now taking an optional
`MarketObserver`, so live trading taps the same stream that is being recorded), plus a
second private WS for `position` / `wallet` / `order`. `StorageWriter` keeps recording
throughout, so a live session also produces training data.

Order state machine, one position at a time:
1. Flat → predictor approves → entry order placed, `orderLinkId` remembered.
2. Resting → the order stream reports `Filled` (position opens at the reported average
   price), reports a terminal status (state cleared), or the deadline passes (cancelled).
3. Open → `decide_exit` each snapshot; on exit a reduce-only order is sent and the
   position is only cleared once the position stream reports flat.

Shutdown order: `ingestion.stop()` → `private_ws.exit()` → drain `ls_ratio_task` and
`trader_task` → drain `storage_task`.

## Telemetry (live)

### Where it goes
`decisions` and `order_events` are written into the same daily SQLite as the market data,
so scoring a session is a join rather than a log parse. Window aggregates are logged to
structlog as `live.rollup` every `cfg.telemetry.rollup_interval_s`.

Volume: ~864k decision rows/day at 100 ms, roughly 130 MB. The `snapshots` table already
writes an order of magnitude more (two 50-level JSON arrays per row), so telemetry adds
about 7% — every decision is kept rather than sampled, which the predicted-vs-realized
join requires anyway. Note the writer commits per row; telemetry roughly doubles the
commit rate, so watch `storage.backpressure` on the first long session.

### The rollup, and which knob each line moves

| Field | Reads as | Knob |
|---|---|---|
| `blocked` | signal policy declined: `flat_selected`, `confidence_below_threshold`, `short_disabled` | `signals.min_confidence`, `signals.allow_short` |
| `risk_rejected` | risk gate declined, by its own nine reasons | `risk.max_spread_bps`, `risk.cooldown_after_trade_ms`, `risk.max_position_notional` |
| `fill_rate`, `cancelled`, `expired` | post-only never traded through / would have crossed | `execution.order_timeout_ms`, quoting side |
| `flow_skew.z_mean_worst` / `z_std_worst` | live features drifted off the frozen training stats | retrain; `features.normalizer_window` |
| `ctx_age_ms` | a context stream went silent behind the forward-fill | halt entries on staleness |
| `loop_ms`, `book_age_ms` | the decision loop is late or the book is stale | `ingestion.snapshot_interval_s`, `risk.stale_data_ms` |
| `class_churn` | argmax flipping between adjacent snapshots — signal is noise | smoothing, `signals.min_confidence` |
| `slippage_bps`, `time_to_fill_ms`, `submit_latency_ms` | the cost of the decision-to-fill delay | exit order type; feeds the latency file |
| `maker_share` | whether the rebate is actually being earned | entry order type |
| `confidence` p50/p99 | is the threshold biting at all | `signals.min_confidence` |
| `inference_us` | model time against the 100 ms budget | `model.seq_len`, device |

### Comparing a live session with a backtest
`PipelineCounters` is one type for both. Back-test the window a session covered, subtract
one funnel from the other, and read the first stage where they disagree: divergence at
`risk_rejected` means the gate saw a different market than modelled, at `fill_rate` means
the queue model is wrong, and at `signals_actionable` means the features differ — check
`flow_skew` first.

### Predicted vs realized
`bot/telemetry/evaluation.py` reads the `decisions` rows that carry a prediction, rebuilds
the label with `compute_labels` (the same smooth mid-price rule, same `alpha`, same
horizon as training) from the `snapshots` that followed, and reports accuracy, per-class
and macro F1, the confusion matrix, predicted-vs-realised class mix and hit rate by
confidence band. Reset zones are excluded exactly as in training.

Because the label rule is shared, the macro F1 here is directly comparable to the
`val_f1_macro` the checkpoint was selected on. A large gap between them is model decay or
feature skew, not an execution problem — check `flow_skew` before touching order handling.
If hit rate does not rise across confidence bands, `min_confidence` is filtering on noise.
