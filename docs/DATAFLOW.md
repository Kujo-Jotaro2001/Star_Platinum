# Star Platinum — полная документация проекта и сквозной поток данных

Документ описывает систему целиком: от байтов, приходящих с Bybit, до ордера,
стоящего в стакане. Каждый шаг привязан к конкретному файлу и функции.

Сопутствующие документы:
- [../CLAUDE.md](../CLAUDE.md) — конституция проекта (стек, конвенции, решения)
- [reference.md](reference.md) — справочник схем БД и кодировок фичей
- **этот файл** — как оно всё соединено и что происходит с данными по дороге

> **Внимание:** [../CLAUDE.md](../CLAUDE.md) местами отстал от кода
> (символ, горизонты, тип exit-ордера). Актуальные значения — в
> [конфигах](#14-конфигурация--полный-справочник) и в разделе
> [Расхождения](#18-инварианты-подводные-камни-и-расхождения).

---

## Оглавление

1. [Что это за система за 60 секунд](#1-что-это-за-система-за-60-секунд)
2. [Карта репозитория](#2-карта-репозитория)
3. [Сквозная схема потока данных](#3-сквозная-схема-потока-данных)
4. [Стадия A — источники данных](#4-стадия-a--источники-данных)
5. [Стадия B — live-приём: WS → SQLite](#5-стадия-b--live-приём-ws--sqlite)
6. [Стадия B′ — исторический загрузчик](#6-стадия-b--исторический-загрузчик-архивы--sqlite--raw-кэш)
7. [Стадия C — оффлайн-пайплайн фичей: SQLite → .npy](#7-стадия-c--оффлайн-пайплайн-фичей-sqlite--npy)
8. [Стадия D — обучение: .npy → checkpoint](#8-стадия-d--обучение-npy--checkpoint)
9. [Стадия E — онлайн-препроцессинг: зеркало стадии C](#9-стадия-e--онлайн-препроцессинг-зеркало-стадии-c)
10. [Стадия F — предсказание → сигнал → риск](#10-стадия-f--предсказание--сигнал--риск)
11. [Стадия G — исполнение: намерение → реальный ордер](#11-стадия-g--исполнение-намерение--реальный-ордер)
12. [Стадия H — бэктест на hftbacktest](#12-стадия-h--бэктест-на-hftbacktest)
13. [Стадия I — телеметрия](#13-стадия-i--телеметрия)
14. [Конфигурация — полный справочник](#14-конфигурация--полный-справочник)
15. [Сквозной пример одной сделки](#15-сквозной-пример-одной-сделки)
16. [Runbook — команды end-to-end](#16-runbook--команды-end-to-end)
17. [Тесты — что чем покрыто](#17-тесты--что-чем-покрыто)
18. [Инварианты, подводные камни и расхождения](#18-инварианты-подводные-камни-и-расхождения)
19. [Текущее состояние данных на диске](#19-текущее-состояние-данных-на-диске)

---

## 1. Что это за система за 60 секунд

ML-бот для бессрочных USDT-фьючерсов Bybit. Одна модель, два пути данных,
общий код принятия решений:

```
             ┌─ оффлайн: архивы/SQLite → .npy → обучение → checkpoint.ckpt
источник ────┤
             └─ онлайн:  WS → те же экстракторы → тот же checkpoint → ордер
```

Ключевая идея всей архитектуры: **экстракторы фичей, политика сигнала, риск-гейт
и жизненный цикл позиции — одни и те же функции** в live, в бэктесте и в
обучении. Онлайн-путь не переписывает препроцессинг — он подменяет только
источник строк (см. [`bot/inference/adapters.py`](../bot/inference/adapters.py)).

Цепочка преобразований данных, в одну строку:

```
WS-сообщение → dataclass → JSON-строка в SQLite → dict → np.ndarray [K,4]/[9]/[17]
  → окно [T,...] → torch.Tensor [B,T,...] → logits [B,H,3] → softmax
  → TradingSignal → RiskDecision → OrderIntent → kwargs pybit → ордер на бирже
  → OpenPosition → ExitDecision → закрывающий ордер → PnL
```

---

## 2. Карта репозитория

### Точки входа (всё через `@hydra.main`)

| Команда | Файл | Что делает |
|---|---|---|
| `python -m bot.main` | [`bot/main.py`](../bot/main.py) | Только приём live-данных в SQLite (Фаза 1) |
| `python -m bot.data.historical_loader` | [`bot/data/historical_loader.py`](../bot/data/historical_loader.py) | Скачать исторические архивы → SQLite + raw-кэш |
| `python -m bot.features.pipeline` | [`bot/features/pipeline.py`](../bot/features/pipeline.py) | SQLite → `.npy` фичи и метки |
| `python -m bot.training.train` | [`bot/training/train.py`](../bot/training/train.py) | Обучение модели → `.ckpt` |
| `python -m bot.backtest.run` | [`bot/backtest/run.py`](../bot/backtest/run.py) | Прогон стратегии на hftbacktest + отчёт |
| `python -m bot.backtest.report <run_dir>` | [`bot/backtest/report.py`](../bot/backtest/report.py) | Перерисовать отчёт без реплея |
| `python -m bot.execution.run` | [`bot/execution/run.py`](../bot/execution/run.py) | Живая торговля (Demo/mainnet) |

### `bot/data/` — приём и хранение (Фаза 1)

| Файл | Роль |
|---|---|
| [`models.py`](../bot/data/models.py) | Пять dataclass'ов рынка: `OrderBookSnapshot`, `Trade`, `TickerContext`, `Liquidation`, `LongShortRatio` |
| [`orderbook.py`](../bot/data/orderbook.py) | `OrderBookManager` — потокобезопасный держатель стакана, детект реконнекта по регрессии `seq` |
| [`ws_client.py`](../bot/data/ws_client.py) | `DataIngestion` — мост pybit-поток → asyncio, таймер снапшотов |
| [`observer.py`](../bot/data/observer.py) | Протокол `MarketObserver` — необязательный live-потребитель того же потока |
| [`storage.py`](../bot/data/storage.py) | `StorageWriter` + все DDL. Асинхронная запись в дневные SQLite |
| [`context_poller.py`](../bot/data/context_poller.py) | Единственный REST-поллер: long/short ratio, раз в 15 мин |
| [`historical_loader.py`](../bot/data/historical_loader.py) | Оффлайн-загрузчик архивов, восстановление книги из дельт |

### `bot/features/` — оффлайн-фичи (Фаза 2)

| Файл | Роль |
|---|---|
| [`replay.py`](../bot/features/replay.py) | Чтение дневных БД в python-dict; `iter_snapshots` — стриминг |
| [`ob_serializer.py`](../bot/features/ob_serializer.py) | Снапшот стакана → тензор `[K,4]` |
| [`flow_features.py`](../bot/features/flow_features.py) | 9 фичей потока сделок на бакет |
| [`context_features.py`](../bot/features/context_features.py) | `ContextBuilder` — 17 контекстных фичей с forward-fill |
| [`labels.py`](../bot/features/labels.py) | Мультигоризонтные метки по сглаженной mid-price |
| [`normalizer.py`](../bot/features/normalizer.py) | `RollingNormalizer` — скользящий z-score, только для flow |
| [`pipeline.py`](../bot/features/pipeline.py) | Оркестратор: всё вышеперечисленное → `.npy` |

### `bot/models/` + `bot/training/` — модель (Фаза 3)

| Файл | Роль |
|---|---|
| [`models/encoders.py`](../bot/models/encoders.py) | `LOBEncoder` (Conv2d), `FlowEncoder` (GRU/Mamba), `CtxEncoder` (MLP) |
| [`models/fusion.py`](../bot/models/fusion.py) | `GatedFusion` — сигмоидный гейт над конкатенацией |
| [`models/hybrid.py`](../bot/models/hybrid.py) | `HybridSignalModel` — энкодеры + фьюжн + голова |
| [`training/dataset.py`](../bot/training/dataset.py) | `LOBDataset` — оконный датасет со `stride` |
| [`training/losses.py`](../bot/training/losses.py) | `MultiHorizonLoss` — CE/focal с маской flat |
| [`training/metrics.py`](../bot/training/metrics.py) | Пер-горизонтные F1 macro и accuracy |
| [`training/train.py`](../bot/training/train.py) | `SignalLitModule` + точка входа Lightning |

### `bot/signals/` + `bot/risk/` — решение (Фаза 4, чистые функции)

| Файл | Роль |
|---|---|
| [`signals/types.py`](../bot/signals/types.py) | `SignalAction`, `SignalPolicy`, `TradingSignal`, `ModelPrediction` |
| [`signals/policy.py`](../bot/signals/policy.py) | `generate_signal` — argmax + порог уверенности |
| [`risk/types.py`](../bot/risk/types.py) | `MarketState`, `PositionState`, `RiskPolicy`, `RiskDecision` |
| [`risk/gate.py`](../bot/risk/gate.py) | `evaluate_risk` — девять проверок в фиксированном порядке |

### `bot/inference/` — онлайн-препроцессинг (Фаза 5)

| Файл | Роль |
|---|---|
| [`buffer.py`](../bot/inference/buffer.py) | `RingBuffer` — скользящее окно `seq_len` |
| [`accumulator.py`](../bot/inference/accumulator.py) | `TradeAccumulator` — сделки между двумя снапшотами |
| [`adapters.py`](../bot/inference/adapters.py) | WS-dataclass → строки, которые ждут экстракторы Фазы 2 |
| [`preprocessor.py`](../bot/inference/preprocessor.py) | `OnlinePreprocessor` — фичи + окна, `is_ready` |
| [`predictor.py`](../bot/inference/predictor.py) | `InferencePredictor` — модель + сигнал + риск в один вызов |
| [`types.py`](../bot/inference/types.py) | `InferenceStatus`, `InferenceResult` |

### `bot/execution/` — исполнение (Фаза 5)

| Файл | Роль |
|---|---|
| [`types.py`](../bot/execution/types.py) | `OrderSide/Type`, `ExitPolicy/Reason`, `OpenPosition`, `OrderIntent`, `InstrumentSpec` |
| [`lifecycle.py`](../bot/execution/lifecycle.py) | Чистые функции: сайзинг входа, TP/SL/hold, эскалация выхода |
| [`orders.py`](../bot/execution/orders.py) | `OrderIntent` → kwargs `place_order` / `cancel_order`, `orderLinkId` |
| [`account.py`](../bot/execution/account.py) | `AccountState` — позиция/эквити/ордера из приватного WS |
| [`live.py`](../bot/execution/live.py) | `LiveTrader` — конечный автомат ордеров, реализует `MarketObserver` |
| [`run.py`](../bot/execution/run.py) | Сборка всего живого стека и запуск |

### `bot/backtest/` — бэктест (Фаза 5)

| Файл | Роль |
|---|---|
| [`prepare.py`](../bot/backtest/prepare.py) | Архивы → feed/snapshot/latency файлы hftbacktest |
| [`strategy.py`](../bot/backtest/strategy.py) | Цикл, который крутит hftbacktest; те же чистые функции |
| [`run.py`](../bot/backtest/run.py) | Скоринг сплита, реплей, метрики, сохранение прогона |
| [`report.py`](../bot/backtest/report.py) | Самодостаточный HTML-отчёт + CLI перерисовки |

### `bot/telemetry/` — наблюдаемость

| Файл | Роль |
|---|---|
| [`types.py`](../bot/telemetry/types.py) | `PipelineCounters` (общий с бэктестом), `DecisionRecord`, `OrderEvent`, статусы |
| [`collector.py`](../bot/telemetry/collector.py) | Счётчики, квантильные окна, `FlowSkew`, роллапы |
| [`evaluation.py`](../bot/telemetry/evaluation.py) | Предсказано vs реализовано, оценка правилом обучающих меток |

---

## 3. Сквозная схема потока данных

```
╔══════════════════════════════ ИСТОЧНИКИ ═══════════════════════════════════╗
║ Bybit WS public      Bybit REST        quote-saver.bycsi.com   Bybit WS     ║
║ orderbook.50 /       /v5/market/       ob500.data.zip          private      ║
║ publicTrade /        account-ratio     + public.bybit.com      position /   ║
║ tickers /            (15 мин)          {SYM}{date}.csv.gz      wallet /     ║
║ allLiquidation                                                 order        ║
╚═══════╤═════════════════╤══════════════════════╤════════════════════╤═══════╝
        │                 │                      │                    │
   ws_client.py     context_poller.py    historical_loader.py    account.py
        │                 │                      │                    │
        ├─────────────────┤                      │                    │
        ▼                 ▼                      ▼                    │
  ┌───────────────────────────┐         ┌──────────────────┐          │
  │ StorageWriter (asyncio    │         │ восстановление   │          │
  │ очередь, drop-oldest)     │         │ книги из дельт   │          │
  └────────────┬──────────────┘         └────────┬─────────┘          │
               ▼                                 ▼                    │
        data/db/{SYM}_{YYYY-MM-DD}.db  ◄─────────┘        data/raw/*.zip,*.gz
        snapshots │ trades │ ticker_context │                    │
        liquidations │ long_short_ratio │                        │
        decisions │ order_events ◄───────────────────────────────┼──┐
               │                                                 │  │
               │ (оффлайн)                                       │ (бэктест)
               ▼                                                 ▼  │
  ┌───────────────────────────────┐              ┌──────────────────────────┐
  │ features/pipeline.py          │              │ backtest/prepare.py      │
  │ ob_serializer + flow + ctx    │              │ bybithistmktdata.convert │
  │ + labels + normalizer         │              │ + eod snapshot + latency │
  └───────────────┬───────────────┘              └────────────┬─────────────┘
                  ▼                                           │
   data/features/{SYM}/*.npy                          data/hft/*.npz
   ob_raw [N,50,4] │ flow [N,9] │ ctx [N,17]                   │
   labels [N,3] │ flat_mask │ timestamps_ms                    │
   normalizer_stats.npz │ top_of_book │ is_reset               │
                  │                                           │
                  ▼                                           │
       ┌──────────────────────┐                               │
       │ training/train.py    │                               │
       │ LOBDataset → модель  │                               │
       └──────────┬───────────┘                               │
                  ▼                                           │
       lightning_logs/**/best-*.ckpt ──────┬──────────────────┤
                  │                        │                  │
      (онлайн)    │                        │ (бэктест)        │
                  ▼                        ▼                  ▼
  ┌───────────────────────────┐   ┌─────────────────────────────────────┐
  │ inference/preprocessor.py │   │ backtest/run.py: скоринг всех окон  │
  │ RingBuffer[50] × 3        │   │ → SignalSchedule (ts_ns → probs)    │
  │ predictor.py → softmax    │   └───────────────┬─────────────────────┘
  └───────────────┬───────────┘                   ▼
                  │                    ┌──────────────────────────┐
                  │                    │ backtest/strategy.py     │
                  │                    │ hbt.elapse(100ms) loop   │
                  ▼                    └───────────────┬──────────┘
      ┌───────────────────────────────────────────────┐│
      │ ОБЩИЙ КОД РЕШЕНИЯ                             ││
      │ signals/policy.generate_signal                │◄┘
      │ risk/gate.evaluate_risk                       │
      │ execution/lifecycle.build_entry_intent        │
      │ execution/lifecycle.decide_exit               │
      └───────────────┬───────────────────┬───────────┘
                      ▼                   ▼
        ┌──────────────────────┐   ┌────────────────────────┐
        │ execution/orders.py  │   │ hbt.submit_buy_order   │
        │ → HTTP.place_order   │   │ (GTX/GTC, LIMIT/MARKET)│
        │ → БИРЖА              │   └──────────┬─────────────┘
        └──────────┬───────────┘              ▼
                   │                  Recorder → record.npz
                   ▼                  → report.html
        приватный WS: fill →
        AccountState → OpenPosition
                   │
                   └──→ telemetry: DecisionRecord + OrderEvent → та же SQLite
```

---

## 4. Стадия A — источники данных

Четыре независимых источника. Ни один не проксируется собственными классами —
pybit и `requests` используются напрямую.

### A.1 Публичный WebSocket (live)

Подписка одним соединением, [`ws_client.py:49-66`](../bot/data/ws_client.py#L49-L66):

| Топик | Callback | Что приносит |
|---|---|---|
| `orderbook.50.{SYMBOL}` | `_on_orderbook` | Полная книга (pybit сам применяет дельты) |
| `publicTrade.{SYMBOL}` | `_on_trade` | Массив сделок: `T`, `i`, `S`, `p`, `v` |
| `tickers.{SYMBOL}` | `_on_ticker` | mark/index price, OI, funding, объёмы |
| `allLiquidation.{SYMBOL}` | `_on_liquidation` | Ликвидации: `T`, `S`, `v`, `p` |

### A.2 REST-поллер (live)

[`context_poller.py`](../bot/data/context_poller.py) — единственный REST-поллер в системе.
`GET /v5/market/account-ratio`, `period="15min"`, `limit=1`, интервал
`cfg.ingestion.ls_ratio_interval_s = 900`. Дедупликация по `timestamp`: строка
пишется только если `ts > last_ts`.

### A.3 Исторические архивы (оффлайн)

[`historical_loader.ensure_raw_files`](../bot/data/historical_loader.py#L182):

| Что | URL | Кэш |
|---|---|---|
| Книга | `{base_url}/orderbook/linear/{SYM}/{date}_{SYM}_ob500.data.zip` | `data/raw/{date}_{SYM}_ob500.data.zip` |
| Сделки | `https://public.bybit.com/trading/{SYM}/{SYM}{date}.csv.gz` | `data/raw/{SYM}{date}.csv.gz` |

Скачивание идёт в `*.part` и переименовывается по завершении — оборванная
загрузка не остаётся в кэше как «готовая». Имена файлов совпадают с
апстримовыми, потому что `hftbacktest.data.utils.bybithistmktdata.convert`
ожидает именно их.

**Важно:** архивы содержат только книгу и сделки. Ни ticker, ни ликвидаций,
ни long/short ratio в них нет — это определяет поведение контекстных фичей
(см. [раздел 18](#18-инварианты-подводные-камни-и-расхождения), пункт 2).

### A.4 Приватный WebSocket (только live-торговля)

[`live.subscribe_private`](../bot/execution/live.py#L148): `position_stream`,
`wallet_stream`, `order_stream` → [`AccountState`](../bot/execution/account.py).
Позиция и эквити никогда не опрашиваются по REST.

---

## 5. Стадия B — live-приём: WS → SQLite

### B.1 Два потока, одна точка стыковки

pybit крутит WS в собственном демон-потоке. Правило в системе одно:

- **Стакан** мутируется прямо из WS-потока, под `threading.Lock`
  ([`OrderBookManager.update`](../bot/data/orderbook.py#L22)).
- **Всё остальное** передаётся в event loop через
  `loop.call_soon_threadsafe(...)` ([`ws_client.py:100`](../bot/data/ws_client.py#L100)).

Поэтому `MarketObserver` (то есть `LiveTrader`) гарантированно вызывается
только на event loop, а не на потоке pybit.

### B.2 Что происходит с одним сообщением стакана

```
pybit callback (WS-поток)
  → _on_orderbook(message)
  → OrderBookManager.update(message["data"])
       seq < last_seq?  → _is_reset = True   (детект реконнекта)
       _data = data                          (сырые списки строк)
```

Само сообщение никуда не пишется. Пишет **таймер**, раз в
`cfg.ingestion.snapshot_interval_s = 0.1` с ([`ws_client.py:69-77`](../bot/data/ws_client.py#L69-L77)):

```
OrderBookManager.snapshot(timestamp_ms=now_ms, depth=50)
  → берёт data["b"][:50], data["a"][:50]
  → Decimal(price), Decimal(qty)             ← строки → Decimal
  → OrderBookSnapshot(ts, bids, asks, is_reset)
  → _is_reset сбрасывается в False (одноразовый флаг)
       │
       ├─→ storage.put_nowait(snap)      → в очередь на запись
       └─→ observer.on_snapshot(snap)    → в LiveTrader, если он есть
```

Таймстемп снапшота — **локальное время машины** (`int(time.time()*1000)`), а не
биржевое. Это и есть та сетка 100 мс, на которой живут все фичи.

### B.3 Сделки, тикер, ликвидации

Разбираются прямо в WS-потоке в замороженные dataclass'ы
([`models.py`](../bot/data/models.py)), затем передаются в loop:

```
_on_trade   → Trade(T, i, S, Decimal(p), Decimal(v))     → _handle_trade
_on_ticker  → TickerContext(11 полей, все Decimal|None)  → _handle_ticker
_on_liq     → Liquidation(T, S, Decimal(v), Decimal(p))  → _handle_liquidation
                     │
                     └→ storage.put_nowait(...) → observer.on_*(...)
```

Порядок фиксирован: **сначала хранилище, потом торговля**. Всё, на чём бот
торговал, гарантированно попало в очередь записи.

### B.4 StorageWriter — запись

[`storage.py`](../bot/data/storage.py):

- Очередь `asyncio.Queue(maxsize=cfg.ingestion.queue_maxsize=10_000)`.
- **Backpressure**: очередь полна → выбрасывается самый старый элемент,
  счётчик `_dropped` логируется на следующем цикле записи
  (`storage.backpressure`).
- **Дневная ротация**: `_ensure_db(timestamp_ms)` считает UTC-дату из
  таймстемпа записи; при смене даты закрывает соединение, открывает
  `data/db/{SYMBOL}_{YYYY-MM-DD}.db` и создаёт все семь таблиц + индексы.
- `await db.commit()` после **каждой** записи.
- Остановка: `run()` выходит из цикла → `_flush()` дочищает очередь → закрытие БД.

### B.5 Схема БД

Семь таблиц в одном дневном файле. Первые пять — рынок, последние две —
телеметрия (пишутся только в live-торговле).

| Таблица | Колонки | Кто пишет |
|---|---|---|
| `snapshots` | `timestamp_ms, is_reset, bids TEXT, asks TEXT` | ingestion + historical |
| `trades` | `timestamp_ms, trade_id, side, price TEXT, qty TEXT` | ingestion + historical |
| `ticker_context` | 11 полей + `collected_at_ms` | только ingestion |
| `liquidations` | `timestamp_ms, side, qty, price, collected_at_ms` | только ingestion |
| `long_short_ratio` | `timestamp_ms UNIQUE, buy_ratio, sell_ratio, collected_at_ms` | только REST-поллер |
| `decisions` | 22 поля — см. [`DecisionRecord`](../bot/telemetry/types.py#L65) | только `LiveTrader` |
| `order_events` | 12 полей — см. [`OrderEvent`](../bot/telemetry/types.py#L99) | только `LiveTrader` |

`bids` / `asks` хранятся как JSON `[["price","qty"], ...]` — строки, не числа,
чтобы `Decimal` пережил round-trip без потерь.

Полные DDL: [`storage.py:25-129`](../bot/data/storage.py#L25-L129).

### B.6 Порядок остановки

[`main.py:66-70`](../bot/main.py#L66-L70): `ingestion.stop()` (закрыть WS) →
отмена задачи ingestion → дождаться `ls_ratio_task` → дождаться `storage_task`
(он дренирует очередь). Именно в этом порядке: иначе часть данных из очереди
пропала бы.

---

## 6. Стадия B′ — исторический загрузчик: архивы → SQLite + raw-кэш

[`historical_loader.py`](../bot/data/historical_loader.py) существует, чтобы
получить те же дневные SQLite-файлы, что даёт live-приём, но за прошлое.
Идемпотентен: день, чей `.db` уже существует, пропускается
([`_load_day`](../bot/data/historical_loader.py#L240)).

### 6.1 Восстановление книги из дельт

Архив `ob500.data.zip` — это поток JSON-строк: первая `type="snapshot"`, все
последующие `type="delta"` с изменившимися уровнями, где `qty="0"` означает
«уровень снят». [`_parse_orderbook_lines`](../bot/data/historical_loader.py#L29)
держит состояние в двух `SortedDict` ровно так же, как `OrderBookManager`
держит его в live — именно это делает оффлайн- и онлайн-книгу идентичными.

```
for i, line in enumerate(lines):
    reset = msg["type"] == "snapshot" or seq < last_seq → очистить обе стороны
    _apply_levels(bids, data["b"]); _apply_levels(asks, data["a"])
    if i % stride != 0: continue          ← прореживание
    yield (ts, is_reset, json(top(bids,50)), json(top(asks,50)))
```

`stride = snapshot_interval_ms // SOURCE_INTERVAL_MS`, где
`SOURCE_INTERVAL_MS = 100`. При `cfg.historical.snapshot_interval_ms = 100`
получается `stride = 1` — сохраняется каждое сообщение.

### 6.2 Разбор CSV сделок

[`_parse_trade_csv`](../bot/data/historical_loader.py#L104). Две ловушки, обе
обработаны явно:

1. Заголовок определяется попыткой `float(row[0])` — **не** `int`, потому что
   Bybit пишет таймстемп как дробные секунды (`"1755648000.1385"`), и `int()`
   принял бы каждую строку данных за заголовок.
2. `ts = int(float(row[0]) * 1000)` — секунды → миллисекунды.
3. `trade_id` берётся из `trdMatchID`, иначе синтезируется как `{ts}-{price}-{qty}`.

### 6.3 Двойное назначение raw-кэша

```
                  ┌→ прорежённый SQLite (100 мс / 50 уровней) → пайплайн фичей
data/raw/*.zip ───┤
                  └→ нетронутый архив (10 мс / 500 уровней)   → hftbacktest
```

Архивы **не удаляются** после конвертации именно поэтому: бэктест кормится
сырьём, а не нашим срезом, иначе он выбросил бы ту самую микроструктуру,
ради которой существует.

---

## 7. Стадия C — оффлайн-пайплайн фичей: SQLite → .npy

[`pipeline.run_pipeline`](../bot/features/pipeline.py#L84) — синхронный код,
никакого event loop.

### 7.1 Оркестрация

```
find_db_files(data_dir, symbol, start, end)      → список дневных .db
count_snapshots(db_paths) → N                    → размер всех выходных массивов
_memmap(...) × 4                                 → ob_raw, flow, ctx, top_of_book
ContextBuilder()                                 → состояние переносится между днями!

для каждого дня:
    day_ts        = load_snapshot_timestamps(db)
    trade_buckets = _bucket_trades(load_trades([db]), day_ts, bucket_ms=100)
    ticker_rows / liq_rows / ls_rows = load_*([db])
    для каждого снапшота (стримингом, iter_snapshots):
        _advance_context(...)  ← «докрутить» контекст до ts снапшота
        ob_raw[j]       = serialize_ob_snapshot(bids, asks, 50)
        top_of_book[j]  = (best_bid, best_ask)
        mid_prices[j]   = (best_bid + best_ask) / 2
        flow_raw[j]     = extract_flow_features(trade_buckets[i])
        ctx_features[j] = ctx_builder.snapshot(ts)
```

Два решения по памяти, оба обязательные:

- **`np.lib.format.open_memmap`** ([`_memmap`](../bot/features/pipeline.py#L74)) —
  один только LOB-тензор весит 800 байт на снапшот, две недели ≈ 10 ГБ.
- **`iter_snapshots`** — день это ~860 тыс. снапшотов по ~20 КБ в python-объектах;
  материализовать даже один день невозможно.

Дни **нельзя** обрабатывать независимо: `ContextBuilder` несёт состояние
(forward-fill, часовое окно OI, минутное окно ликвидаций) через границу суток.

### 7.2 LOB-тензор `[K,4]`

[`serialize_ob_snapshot`](../bot/features/ob_serializer.py#L6). Для уровня `k`:

| Колонка | Формула |
|---|---|
| 0 | `(best_bid - bid_price[k]) / best_bid` — относительный отступ от лучшей цены |
| 1 | `log1p(bid_qty[k])` |
| 2 | `(ask_price[k] - best_ask) / best_ask` |
| 3 | `log1p(ask_qty[k])` |

Кодировка **самонормирующаяся** — z-score к ней не применяется. Недостающие
уровни добиваются нулями. Абсолютной цены в тензоре нет — поэтому и существует
`top_of_book.npy`.

### 7.3 Flow-фичи `[9]`

[`extract_flow_features`](../bot/features/flow_features.py#L12), на бакет
`[T - 100 мс, T)`:

| # | Фича | Формула |
|---|---|---|
| 0 | `buy_volume` | `log1p(сумма qty по Buy)` |
| 1 | `sell_volume` | `log1p(сумма qty по Sell)` |
| 2 | `signed_volume` | `log1p(abs(buy-sell)) * sign(buy-sell)` |
| 3 | `trade_count` | число сделок |
| 4 | `ofi` | `(buy-sell) / (buy+sell+1e-8)` |
| 5 | `max_trade_size` | `log1p(max qty)` |
| 6 | `mean_trade_size` | `log1p(total/n)` |
| 7 | `buy_count` | целое |
| 8 | `sell_count` | целое |

Пустой бакет → вектор нулей. Привязка бакета к снапшоту делается
`bisect` по отсортированным таймстемпам сделок
([`_bucket_trades`](../bot/features/pipeline.py#L28)).

### 7.4 Контекстные фичи `[17]`

[`ContextBuilder`](../bot/features/context_features.py#L15). Индексы:

| # | Фича | Источник |
|---|---|---|
| 0–1 | `mark_price`, `index_price` | ticker, forward-fill |
| 2 | `basis_bps` = `(mark-index)/index*10000` | вычисляется |
| 3 | `open_interest` | ticker |
| 4 | `oi_change_pct_1h` | окно `oi_history_window_ms = 3_600_000` |
| 5–6 | `funding_rate`, `funding_sign` | ticker |
| 7–8 | `volume_24h`, `price_24h_pct` | ticker |
| 9–12 | `liq_buy_vol`, `liq_sell_vol`, `liq_imbalance`, `liq_count` | окно `liq_window_ms = 60_000` |
| 13–15 | `ls_buy_ratio`, `ls_sell_ratio`, `ls_ratio` | REST-поллер |
| 16 | `open_interest_value` | ticker |

Контекст **не нормализуется** и forward-fill'ится бесконечно. Именно поэтому
рядом живёт [`source_ages_ms`](../bot/features/context_features.py#L92):
замолчавший стрим неотличим от тихого, если смотреть только на фичи.

### 7.5 Метки

[`compute_labels`](../bot/features/labels.py#L4), метод «сглаженная mid-price»:

```
prev = mean(mid[t-h : t])
fut  = mean(mid[t : t+h])
ret  = (fut - prev) / prev
ret >  alpha → 2 (up)
ret < -alpha → 0 (down)
иначе        → 1 (flat), flat_mask = True
```

Окна считаются через префиксные суммы — иначе при `h = 300` и `N` порядка
миллионов это квадратичная операция, доминирующая над всем пайплайном.

`valid_mask = False` для:
- зон реконнекта: `max_h` снапшотов по обе стороны от `is_reset`;
- краёв, где широчайшему горизонту не хватает места: `[:max_h]` и `[n-max_h+1:]`;
- строк, где `prev == 0`.

Затем в пайплайне ([`pipeline.py:176-177`](../bot/features/pipeline.py#L176-L177)):
```python
labels[~valid_mask] = 1
flat_mask[~valid_mask] = True
```
Невалидные строки маршрутизируются через ту же маску `flat`, которую лосс и
метрики уже исключают. Без этого они по умолчанию несли бы метку `0` и модель
училась бы на них как на уверенном «вниз».

### 7.6 Нормализация

[`RollingNormalizer`](../bot/features/normalizer.py), **только для flow-фичей**:

```
train_end = int(N * train_ratio)
for i in range(train_end):     flow_raw[i] = norm.update_and_normalize(flow_raw[i])
for i in range(train_end, N):  flow_raw[i] = norm.normalize_only(flow_raw[i])
norm.save(output_dir / "normalizer_stats.npz")
```

Нормализация делается **на месте**, прямо в memmap: файл `flow_features.npy`
уже z-scored, и обучение с бэктестом читают его как есть.

> ⚠️ Тонкость: `_recompute_stats` считает mean/std по deque'у последних
> `normalizer_window = 500` значений. Значит, «замороженная статистика
> обучающего сплита» — это статистика **последних 500 строк** трейна (≈50 секунд),
> а не всего сплита. Онлайн-путь потом z-scored'ит именно по ней. См.
> [раздел 18](#18-инварианты-подводные-камни-и-расхождения), пункт 5.

### 7.7 Выход стадии C

`data/features/{SYMBOL}/`:

| Файл | Форма | dtype | Что это |
|---|---|---|---|
| `ob_raw.npy` | `[N, 50, 4]` | float32 | LOB-тензор |
| `flow_features.npy` | `[N, 9]` | float32 | Flow, **z-scored** |
| `ctx_features.npy` | `[N, 17]` | float32 | Контекст, сырой |
| `labels.npy` | `[N, 3]` | int8 | 0=down, 1=flat, 2=up |
| `flat_mask.npy` | `[N, 3]` | bool | True = исключить из лосса |
| `valid_mask.npy` | `[N]` | bool | Диагностика (уже вшита в labels) |
| `timestamps_ms.npy` | `[N]` | int64 | UTC unix ms |
| `is_reset.npy` | `[N]` | bool | Снапшот после реконнекта |
| `top_of_book.npy` | `[N, 2]` | float64 | Абсолютные best_bid/best_ask |
| `normalizer_stats.npz` | — | — | mean/std/buffers для онлайна |

`K = cfg.ingestion.lob_depth = 50`, `H = len(cfg.features.label_horizons) = 3`.

---

## 8. Стадия D — обучение: .npy → checkpoint

### 8.1 Датасет

[`LOBDataset`](../bot/training/dataset.py). Окно `seq_len = 50` подряд идущих
снапшотов (= 5 секунд), метка берётся с **последнего** таймстепа окна.

```
start = idx * stride        # stride = cfg.train.window_stride = 10
end   = start + seq_len
ob    → [50, 50, 4]
flow  → [50, 9]
ctx   → [50, 17]   (или нули, если use_ctx=False)
labels    → [3]  int64
flat_mask → [3]  bool
```

`stride = 10` — соседние окна на сетке 100 мс перекрываются более чем на 98%,
поэтому обучение на каждом стоило бы порядок величины компьюта почти без
прироста информации. При `stride=10` получается одно обучающее окно в секунду.

Разбиение хронологическое, по строкам, **до** построения датасета
([`train.py:114-131`](../bot/training/train.py#L114-L131)) — окна не пересекают
границы сплитов:

```
train: [0, 0.8N)   val: [0.8N, 0.9N)   test: [0.9N, N)
```

### 8.2 Модель

[`HybridSignalModel`](../bot/models/hybrid.py):

```
ob   [B,50,50,4] ─→ LOBEncoder  ─→ [B,50,64] ┐
flow [B,50,9]    ─→ FlowEncoder ─→ [B,50,64] ├→ GatedFusion → [B,50,64]
ctx  [B,50,17]   ─→ CtxEncoder  ─→ [B,50,64] ┘        │
                                                       ▼
                                              mean по T → [B,64]
                                                       ▼
                                        Linear(64 → H*3) → [B,3,3] логиты
```

| Компонент | Устройство |
|---|---|
| [`LOBEncoder`](../bot/models/encoders.py#L6) | `permute → [B,4,T,K]`, `n_lob_blocks=2` блоков `Conv2d+BatchNorm+GELU+Dropout2d`, ядра `(1,3)` затем `(3,3)`, `AdaptiveAvgPool2d((None,1))` схлопывает K |
| [`FlowEncoder`](../bot/models/encoders.py#L68) | GRU (`gru_layers=2`, batch_first) или Mamba по опции `cfg.model.flow_encoder="mamba"` |
| [`CtxEncoder`](../bot/models/encoders.py#L118) | `Linear(17→64) + GELU + LayerNorm`, по таймстепам |
| [`GatedFusion`](../bot/models/fusion.py) | `concat → candidate(Linear+GELU)`, `gate(Linear+Sigmoid)`, результат `gate * candidate` |

`use_ctx` фиксируется в конструкторе (а не в `forward`) ради чистого экспорта в ONNX.

### 8.3 Лосс и метрики

[`MultiHorizonLoss`](../bot/training/losses.py):
```
ce = cross_entropy(logits[B*H,3], targets[B*H], weight=class_weights, reduction="none")
если focal:  ce *= (1 - p_t)^gamma
loss = (ce.view(B,H) * ~flat_mask).sum() / (~flat_mask).sum()
```

Все `H` горизонтов обучаются совместно. Веса классов — обратно-частотные,
посчитанные по не-flat меткам **только трейна**
([`_compute_class_weights`](../bot/training/train.py#L86)).

Валидация: [`compute_metrics`](../bot/training/metrics.py) — per-horizon
`f1_h{h}`, `acc_h{h}` и усреднённый `f1_macro`, всё с той же маской.

### 8.4 Тренер

- AMP `16-mixed` при наличии CUDA, иначе `32-true`.
- `ModelCheckpoint(monitor=cfg.train.val_metric="val_f1_macro", mode="max", save_top_k=1)`.
- `EarlyStopping(patience=3)`, `gradient_clip_val=1.0`, `deterministic=True`.
- Так как `hydra.run.dir: .`, чекпоинты ложатся в
  `lightning_logs/version_{k}/checkpoints/best-epoch=..-val_f1_macro=...ckpt`.

Чекпоинт Lightning содержит `state_dict` с префиксом `model.`; его снимает
[`_model_state_dict`](../bot/inference/predictor.py#L201) при загрузке —
поэтому и live, и бэктест грузят `.ckpt` напрямую, без Lightning.

---

## 9. Стадия E — онлайн-препроцессинг: зеркало стадии C

Это место, где онлайн и оффлайн обязаны совпасть до бита.

### 9.1 Адаптеры

[`adapters.py`](../bot/inference/adapters.py) — единственный «переводчик»:
WS-dataclass → dict той же формы, что читается из SQLite.

| Функция | Из | В |
|---|---|---|
| `book_levels(snapshot)` | `OrderBookSnapshot` | `([["price","qty"],...], [...])` для `serialize_ob_snapshot` |
| `trade_row(trade)` | `Trade` | `{side, price, qty}` для `extract_flow_features` |
| `ticker_row(ticker)` | `TickerContext` | строка `ticker_context` для `ContextBuilder.update_ticker` |
| `liquidation_row(liq)` | `Liquidation` | строка `liquidations` |
| `long_short_row(ratio)` | `LongShortRatio` | строка `long_short_ratio` |

Благодаря этому **экстракторы Фазы 2 вызываются буквально те же самые** —
никакой второй реализации фичей не существует.

### 9.2 OnlinePreprocessor

[`preprocessor.py`](../bot/inference/preprocessor.py). На каждый снапшот:

```
append_snapshot(ts, bids, asks, trades):
    ob   = serialize_ob_snapshot(bids, asks, ob_depth)      ← тот же код
    flow = normalizer.normalize_only(extract_flow_features(trades))
                                                            ← замороженная статистика
    ctx  = ctx_builder.snapshot(ts)                         ← тот же ContextBuilder
    ob_buffer.append(ob); flow_buffer.append(flow); ctx_buffer.append(ctx)
```

Три [`RingBuffer`](../bot/inference/buffer.py) ёмкости `seq_len = 50`.
`is_ready` = все три полны, то есть после 50 снапшотов ≈ 5 секунд разогрева.
`window()` разворачивает кольцо в хронологический порядок.

`normalize_only` (а не `update_and_normalize`) — принципиально: онлайн **не**
обновляет статистику, иначе распределение уплывало бы от того, на котором
училась модель.

### 9.3 Разница онлайна и оффлайна в flow-бакете

| | Оффлайн | Онлайн |
|---|---|---|
| Границы бакета | строго `[T - 100 мс, T)` по таймстемпам сделок | всё, что накопил [`TradeAccumulator`](../bot/inference/accumulator.py) с предыдущего снапшота |

Если петля просела и снапшоты пришли с интервалом 180 мс, live-бакет содержит
180 мс сделок против обучающих 100 мс. Это ровно тот класс расхождения, который
ловит `FlowSkew` (см. [раздел 13](#13-стадия-i--телеметрия)).

---

## 10. Стадия F — предсказание → сигнал → риск

### 10.1 Предсказание

[`InferencePredictor.predict`](../bot/inference/predictor.py#L107):

```
if not preprocessor.is_ready: → InferenceResult(NOT_READY, "buffers_not_ready")

ob, flow, ctx = preprocessor.windows()
tensors: [1,50,50,4], [1,50,9], [1,50,17]  → .to(device)
with torch.inference_mode():
    logits = model(ob, flow, ctx if use_ctx else None)   → [1,3,3]
    probs  = softmax(logits, -1)[0]                      → [3,3]
```

`probabilities` — кортеж из `H` троек `(p_down, p_flat, p_up)`.

### 10.2 Политика сигнала

[`generate_signal`](../bot/signals/policy.py#L22), на горизонте
`cfg.signals.horizon_index = 1` (то есть 100 снапшотов = 10 с):

```
selected = argmax(class_probs[horizon_index]);  confidence = p[selected]

selected == FLAT               → NO_TRADE, reason="flat_selected"
confidence < min_confidence    → NO_TRADE, reason="confidence_below_threshold"
selected == UP                 → BUY
selected == DOWN and allow_short → SELL
иначе                          → NO_TRADE, reason="short_disabled"
```

У каждого исхода есть `reason` — это то, что позже считается по имени в
`PipelineCounters.signal_blocked`.

### 10.3 Риск-гейт

[`evaluate_risk`](../bot/risk/gate.py#L9) — девять проверок, порядок значим
(первая сработавшая и есть причина отказа):

| # | Проверка | `reason` |
|---|---|---|
| 1 | Действие не BUY/SELL | `signal_not_actionable` |
| 2 | Символ сигнала ≠ символ позиции/рынка | `symbol_mismatch` |
| 3 | `now_ms - market.timestamp_ms > stale_data_ms` | `stale_market_data` |
| 4 | `spread_bps > max_spread_bps` | `spread_too_wide` |
| 5 | `confidence < min_confidence` | `confidence_below_threshold` |
| 6 | SELL при `allow_short=False` | `short_disabled` |
| 7 | С последней сделки прошло `< cooldown_after_trade_ms` | `cooldown_active` |
| 8 | Итоговый нотионал `> max_position_notional` | `max_position_notional_exceeded` |
| 9 | `notional / equity > max_leverage` | `max_leverage_exceeded` |
| — | всё пройдено | `approved` |

Итоговый нотионал считается **знаково**
([`_resulting_notional`](../bot/risk/gate.py#L70)): `abs(позиция*mid ± ордер)` —
закрывающий ордер уменьшает экспозицию, а не увеличивает её.

Результат — [`InferenceResult`](../bot/inference/types.py) со статусом
`NOT_READY` / `APPROVED` / `REJECTED`.

---

## 11. Стадия G — исполнение: намерение → реальный ордер

[`LiveTrader`](../bot/execution/live.py) — конечный автомат с ровно тремя
состояниями: **свободен** → **ждём исполнения входа** (`_resting`) →
**в позиции** (`_position`).

### 11.1 Почему снапшоты кладутся в очередь

`on_snapshot` не торгует, а кладёт снапшот в `asyncio.Queue(maxsize=100)` с
drop-oldest ([`live.py:129-132`](../bot/execution/live.py#L129-L132)). Шаг
принятия решения `await`-ит блокирующие REST-вызовы pybit через
`asyncio.to_thread`, и не должен тормозить приём данных.

### 11.2 Один цикл `step()`

[`live.step`](../bot/execution/live.py#L164):

```
now_ms = локальное время
если книга пуста        → record(NO_BOOK) → выход
frame = preprocessor.append_snapshot(ts, bids, asks, accumulator.drain())
market = MarketState(symbol, ts, best_bid, best_ask, account.equity)

если есть позиция       → _manage_position()   → record(HOLDING)
иначе если есть resting → _reconcile_resting() → record(WAITING)
иначе                   → _maybe_enter()       → record(APPROVED|REJECTED|BLOCKED|NOT_READY)

_maybe_rollup(now_ms)
```

**Каждый** проход пишет `DecisionRecord` — включая те, где предсказания не было.
Как часто петля вообще не может действовать — это диагностика данных и разогрева,
и она невидима, если логировать только сделки.

### 11.3 Вход

[`_maybe_enter`](../bot/execution/live.py#L351) → [`build_entry_intent`](../bot/execution/lifecycle.py#L19):

```
side            = BUY | SELL из сигнала
reference_price = PostOnly ? пассивная сторона : пересекающая сторона
                  (BUY PostOnly → best_bid; BUY Market → best_ask)
qty             = floor_to_step(order_notional / reference_price, qty_step)
если qty < min_order_qty → None (ордер не выставляется)
limit_price     = reference_price для PostOnly, иначе None
```

Сайзинг идёт от цены, **по которой ордер реально торгуется**, — так что нотионал,
который одобрил риск-гейт, и есть принятый на себя нотионал.

Далее [`place_order_params`](../bot/execution/orders.py#L8) превращает намерение
в kwargs pybit:

```python
{"category": "linear", "symbol": "SOLUSDT", "side": "Buy",
 "orderType": "Limit",            # PostOnly — это Limit + timeInForce, не свой тип
 "qty": "0.5", "reduceOnly": False,
 "orderLinkId": "sp-e-b-1756423891234",
 "price": "182.34", "timeInForce": "PostOnly"}
```

- Цены и количества форматируются ровно с точностью `price_tick` / `qty_step`
  ([`format_decimal`](../bot/execution/orders.py#L45)) — лишние знаки Bybit отклоняет.
- `orderLinkId = sp-{e|x}-{b|s}-{created_ts_ms}` — клиентский ID, по которому
  апдейты из приватного WS сопоставляются с нашими намерениями.
- При `cfg.execution.dry_run=true` вызов `place_order` **не делается**,
  но `OrderEvent("submitted")` всё равно пишется — это и есть режим первого
  прогона на Demo.

Ордер запоминается как
`_RestingOrder(intent, link_id, deadline_ms = now + order_timeout_ms, submitted_ms, decision_price)`.

### 11.4 Согласование резервного ордера

[`_reconcile_resting`](../bot/execution/live.py#L282) на каждом снапшоте
смотрит в [`AccountState.order(link_id)`](../bot/execution/account.py#L98) —
состояние, которое приватный WS кладёт из потока pybit под тем же `threading.Lock`:

| Состояние ордера | Действие | `OrderEvent` |
|---|---|---|
| `Filled` | `open_position(...)` по **реальной** `avg_price` | `filled` |
| Не открыт и не исполнен (`Cancelled`/`Rejected`) | сбросить | `expired` |
| Ещё висит, но `now_ms >= deadline_ms` | `cancel_order` | `cancelled` |

Даже рыночный вход проходит через это же ожидание: скобку TP/SL надо строить на
фактической средней цене исполнения, а не на цене решения.

### 11.5 Открытие позиции и скобка выхода

[`open_position`](../bot/execution/lifecycle.py#L53) + [`exit_prices`](../bot/execution/lifecycle.py#L72):

```
BUY:  TP = fill*(1 + tp_bps/10000)   SL = fill*(1 - sl_bps/10000)
SELL: TP = fill*(1 - tp_bps/10000)   SL = fill*(1 + sl_bps/10000)
hold_deadline_ms = fill_ts + hold_ms
```

TP/SL — **триггерные уровни, а не цены ордеров**, поэтому они намеренно не
округляются к сетке тика: сам выход котируется от живой книги.

`hold_ms = hold_snapshots * bucket_ms`, где `hold_snapshots` по умолчанию равен
`cfg.features.label_horizons[cfg.signals.horizon_index]` — то есть удержание
равно тому горизонту, на котором модель обучалась. Это и делает бэктест
сравнимым с валидационной метрикой.

### 11.6 Выход

[`decide_exit`](../bot/execution/lifecycle.py#L96) — приоритет фиксирован:

```
exit_price = BUY ? best_bid : best_ask       ← цена, по которой можно выйти
1. стоп-лосс пробит      → STOP_LOSS
2. тейк-профит достигнут → TAKE_PROFIT
3. now >= hold_deadline  → HOLD_EXPIRED
иначе                    → не выходим
```

Стоп бьёт тейк, когда один снапшот перекрывает оба уровня, и оба бьют истечение
удержания.

Далее ([`_manage_position`](../bot/execution/live.py#L205)):

```
build_exit_intent(reduce_only=True, exit_order_type)  → пассивный лимит
_submit(...) → _exit_pending = True
на следующих снапшотах:
    account.is_flat?                 → позиция закрыта, OrderEvent("filled") с exit_reason
    иначе, если ещё не эскалировали  → _escalate_exit()
```

[`should_cross_exit`](../bot/execution/lifecycle.py#L143) решает, когда бросать
пассивную попытку:

```
exit_fallback_ms <= 0                      → никогда
now - submitted >= exit_fallback_ms        → пересекать
стоп-лосс пробит, пока ордер стоял         → пересекать немедленно
```

Защитный выход, который не исполняется, — не защитный. Поэтому пассивная
попытка получает своё окно (`exit_fallback_ms = 2000`) и затем отменяется
в пользу маркета.

### 11.7 Сборка живого стека

[`execution/run.py`](../bot/execution/run.py) — что с чем соединяется:

```
InferencePredictor.from_checkpoint(ckpt, normalizer_stats, политики, гиперпараметры модели)
AccountState(symbol)                    ← приватный WS
StorageWriter(data_dir, symbol)         ← та же дневная SQLite, что и у ingestion
LiveTrader(...)                         ← observer
WebSocket(channel_type="private", demo) → trader.subscribe_private(ws)
DataIngestion(..., observer=trader)     ← публичный WS
poll_long_short_ratio(..., observer=trader)

задачи: storage.run | ingestion.run | trader.run | ls_ratio poller
```

Ключи API берутся только из окружения: `BYBIT_API_KEY` / `BYBIT_API_SECRET`
(через `python-dotenv`), никогда из YAML.

---

## 12. Стадия H — бэктест на hftbacktest

Собственного движка нет: он был удалён, чтобы существовала одна модель
исполнения и один набор цифр.

### 12.1 Подготовка фида

[`prepare.prepare_range`](../bot/backtest/prepare.py#L145):

```
convert_day            → bybithistmktdata.convert(ob500.zip, trades.csv.gz)
                         → data/hft/{SYM}_{date}.npz
build_order_latency    → generate_order_latency(mul_entry, offset_entry, ...)
                         → data/hft/{SYM}_{date}_latency.npz
_previous_day_snapshot → create_last_snapshot(...) → {SYM}_{date}_eod.npz
```

EOD-снапшот предыдущего дня нужен, чтобы первый день не открывался на пустой
книге. Он строится **только из уже скачанных** архивов: день до запрошенного
диапазона может выходить за пределы доступной истории, и незапрошенная
догрузка была бы сюрпризом.

### 12.2 Согласование расписания

Модель живёт на сетке 100 мс, фид — потиковый. Поэтому предсказания
вычисляются заранее и привязываются к таймстемпам снапшотов
([`build_schedule`](../bot/backtest/run.py#L65)):

```
окно j датасета покрывает [j, j+seq_len) и размечено на последнем шаге
⇒ строка j соответствует снапшоту (seq_len - 1 + j)
timestamps_ns = timestamps_ms[seq_len-1 : ...] * 1_000_000
```

В цикле [`_advance`](../bot/backtest/strategy.py#L317) берёт **самое свежее уже
опубликованное** предсказание — заглянуть в будущее структурно невозможно.

Фид покрывает целые сутки, а сплит — нет. За последним предсказанием
(`past_end`) стратегия перестаёт открываться и выходит, как только становится
плоской ([`strategy.py:105-110`](../bot/backtest/strategy.py#L105-L110)).

### 12.3 Цикл стратегии

[`run_strategy`](../bot/backtest/strategy.py#L60), шаг
`elapse_ns = bucket_ms * 1e6 = 100 мс`:

```
while hbt.elapse(100ms) == 0:
    recorder.record(hbt)
    best_bid, best_ask = _top_of_book(hbt.depth(0))   ← цены из тиков, не из float
    cursor = _advance(schedule.timestamps_ns, cursor, now_ns)
    market = MarketState(..., account_equity=_equity(hbt, initial_equity, ...))

    resting?  → проверить fill / отмену / дедлайн
    position? → exit_pending ? эскалация : decide_exit → выставить выход
    иначе     → generate_signal → evaluate_risk → build_entry_intent → submit
    hbt.clear_inactive_orders(0)
```

Три неочевидные детали, каждая с комментарием в коде:

1. [`_top_of_book`](../bot/backtest/strategy.py#L276) восстанавливает цену из
   **тиков** (`best_bid_tick * tick_size`): `depth.best_ask` — float и приходит
   как `100.10000000000001`, что отравило бы всю `Decimal`-арифметику.
   Пустая сторона у hftbacktest — `NaN`.
2. [`_equity`](../bot/backtest/strategy.py#L294) прибавляет
   `cfg.backtest.initial_equity`: у hftbacktest нет понятия стартового баланса,
   `state_values.balance` — это денежный поток от торговли и открывается нулём.
   Иначе проверка плеча делила бы на пустой счёт.
3. [`_submit`](../bot/backtest/strategy.py#L252): post-only в hftbacktest — это
   `GTX` (ордер, который пересёк бы книгу, истекает неисполненным), а не
   отдельный тип.

Стратегия намеренно **не** `@njit` — именно это позволяет вызывать те же
`signals` / `risk` / `execution.lifecycle`, что и `LiveTrader`.

### 12.4 Модели очереди и латентности

[`build_asset`](../bot/backtest/run.py#L88):

| Настройка | Значение по умолчанию |
|---|---|
| `queue_model` | `power_prob3` с `queue_power=3.0` |
| `latency_model` | `intp` — интерполяция по сгенерированным latency-файлам |
| `maker_fee_rate` | `-0.0001` (ребейт) |
| `taker_fee_rate` | `0.00055` |
| Заполнение | `no_partial_fill_exchange()` |

> ⚠️ **Оговорка о латентности.** У публичной истории Bybit нет локального
> таймстемпа, поэтому задержка фида **синтезируется** прибавлением
> `feed_latency_ns = 10 мс` к каждому биржевому таймстемпу, а задержка ордеров
> выводится из неё линейно. Она структурирована, но не измерена. `RunMeta`
> несёт `latency_is_measured=False`, и отчёт это показывает. Сессия на Demo с
> телеметрией `OrderEvent.latency_ms` — это то, что превращает
> `intp_order_latency` из догадки в данные.

### 12.5 Отчёт

[`save_run`](../bot/backtest/report.py#L227) пишет в
`{report_dir}/{SYMBOL}_{split}/`:

| Файл | Содержимое |
|---|---|
| `record.npz` | Сырая запись `Recorder` (timestamp, price, balance, position, fee) |
| `run.json` | `summary` (метрики hftbacktest), `strategy` (`PipelineCounters`), `meta` (`RunMeta`) |
| `report.html` | Самодостаточная страница |

`python -m bot.backtest.report <run_dir>` перерисовывает HTML из этих файлов без
повторного реплея ([`load_run`](../bot/backtest/report.py#L249)).

Кривые ([`equity_curves`](../bot/backtest/report.py#L71)) — эквити, эквити без
комиссий и buy-and-hold — все приведены к процентам от размера книги, чтобы
делить одну ось. Собственный график hftbacktest кладёт цену на вторую ось, что
глазами не читается. Прореживание
([`downsample`](../bot/backtest/report.py#L97)) сохраняет **максимальную**
просадку в каждом бакете: простой шаг выбросил бы именно те пики, ради которых
смотрят underwater-график.

Воронка решений (actionable → approved → submitted → filled/cancelled/expired) —
наша, не hftbacktest'а. Именно она говорит, чья вина в слабом результате:
модели или исполнения.

---

## 13. Стадия I — телеметрия

Каждый показатель отвечает на вопрос «какая стадия сломалась», чтобы плохая
сессия указывала на ручку, а не на «бот потерял деньги».

### 13.1 Одна воронка для live и бэктеста

[`PipelineCounters`](../bot/telemetry/types.py#L19) — **общий тип**. Прогоните
бэктест на окне, которое покрыла живая сессия, вычтите одну воронку из другой:
стадия, где они разошлись, и есть ошибка моделирования. Разные типы сделали бы
это сравнение гаданием.

`signal_blocked` и `risk_rejected` — словари по именам причин, которые уже
возвращают политика сигнала и риск-гейт.

### 13.2 Статусы прохода петли

[`types.py:7-15`](../bot/telemetry/types.py#L7-L15). Первые четыре означают, что
предсказания вообще не было:

| Статус | Смысл |
|---|---|
| `NO_BOOK` | Пустая книга |
| `NOT_READY` | Кольцевые буферы ещё не заполнены (разогрев) |
| `HOLDING` | Мы в позиции, вход не рассматривался |
| `WAITING` | Висит неисполненный вход |
| `BLOCKED` | Политика сигнала отказалась (`flat`, низкая уверенность, шорт выключен) |
| `REJECTED` | Риск-гейт отклонил |
| `APPROVED` | Ордер выставлен |

`BLOCKED` и `REJECTED` разделены умышленно
([`_status_of`](../bot/execution/live.py#L492)): это разные отказы и они не
должны делить один счётчик.

### 13.3 Train/serve skew — первое, что нужно смотреть

[`FlowSkew`](../bot/telemetry/collector.py#L59). `OnlinePreprocessor` z-scored'ит
по mean/std, замороженным из трейна, значит живой поток, совпадающий с обучением,
даёт `z ~ N(0,1)`:

- `z_mean_worst` далеко от 0 → сдвиг уровня;
- `z_std_worst` далеко от 1 → взрыв дисперсии.

Никакая настройка исполнения не чинит того, что модели задают вопрос о
распределении, которого она не видела.

### 13.4 Прочие датчики

| Метрика | Что диагностирует |
|---|---|
| `loop_ms` p50/p99/max | Держит ли петля сетку 100 мс |
| `book_age_ms` | Насколько устарел стакан к моменту решения |
| `inference_us` | Стоимость форварда |
| `confidence` | Распределение уверенности — калибровка `min_confidence` |
| `time_to_fill_ms`, `submit_latency_ms` | То самое, что бэктест только предполагает |
| `slippage_bps` | Знаковое: положительное всегда «хуже задуманного» |
| `ctx_age_ms` (ticker/liq/ls) | Замолчавший источник контекста |
| `class_churn` | Как часто меняется предсказанный класс |
| `maker_share` | Доля исполнений по пассивной цене |

Роллап логируется через `structlog` каждые `cfg.telemetry.rollup_interval_s = 60`
и сбрасывает оконные счётчики ([`rollup`](../bot/telemetry/collector.py#L223)).

### 13.5 Предсказано vs реализовано

[`evaluation.py`](../bot/telemetry/evaluation.py) — намеренно переиспользует
[`compute_labels`](../bot/features/labels.py): живое предсказание оценивается тем
же правилом сглаженной mid-price, которым размечался трейн, поэтому получившееся
число **напрямую сравнимо** с валидационной F1, по которой выбирали чекпоинт.
Любая другая формула дала бы метрику, похожую на F1 и несравнимую с ней.

```
load_decisions(db_paths)   → строки decisions со статусом из PREDICTED_STATUSES
load_mid_prices(db_paths)  → mid из таблицы snapshots
score(...)                 → LiveScore(accuracy, f1_macro, f1_per_class,
                             confusion, hit_rate_by_confidence,
                             predicted_mix, realised_mix)
```

`hit_rate_by_confidence` — точность по полосам уверенности. Если она не растёт с
уверенностью, порог `min_confidence` фильтрует шум.

### 13.6 Где это лежит

`decisions` и `order_events` пишутся в **ту же дневную SQLite**, что и рыночные
данные, поэтому скоринг — это просто join. Строка на каждое решение
(≈864 тыс./день, ~130 МБ) пишется без сэмплирования: таблица `snapshots` и так
пишет на порядок больше, а сэмплирование сломало бы join «предсказано vs
реализовано».

---

## 14. Конфигурация — полный справочник

Правила ([`CLAUDE.md`](../CLAUDE.md)): всё настраиваемое живёт в `conf/` как
обычный YAML. Никаких типизированных dataclass-конфигов. `main(cfg)` достаёт
значения и передаёт их явно; внутренние функции `cfg` не видят.

### [`conf/config.yaml`](../conf/config.yaml)

`defaults` со всеми группами; `hydra.run.dir: .` (иначе Hydra уводит cwd в
`outputs/` и ломает относительные пути), `hydra.output_subdir: null`.

### [`conf/ingestion/default.yaml`](../conf/ingestion/default.yaml)

| Ключ | Значение | Смысл |
|---|---|---|
| `symbol` | `SOLUSDT` | Единственный символ MVP |
| `data_dir` | `data/db` | Куда пишутся дневные SQLite |
| `lob_depth` | `50` | **Единственный источник истины для K** |
| `snapshot_interval_s` | `0.1` | Сетка 100 мс |
| `queue_maxsize` | `10000` | Граница очереди `StorageWriter` |
| `ls_ratio_interval_s` | `900` | Интервал REST-поллера |

### [`conf/historical/default.yaml`](../conf/historical/default.yaml)

`start_date` / `end_date` (`2025-08-07` … `2025-08-20`), `base_url`
(`https://quote-saver.bycsi.com`), `snapshot_interval_ms: 100`, `raw_dir: data/raw`.

### [`conf/features/default.yaml`](../conf/features/default.yaml)

| Ключ | Значение | Смысл |
|---|---|---|
| `start_date` / `end_date` | `2025-08-07` / `2025-08-20` | Диапазон обработки |
| `bucket_ms` | `100` | Ширина бакета сделок |
| `train_ratio` / `val_ratio` | `0.8` / `0.1` | Хронологическое 80/10/10 |
| `normalizer_window` | `500` | Окно скользящего z-score |
| `label_alpha` | `0.0001` | 1 bps — даёт ~31/38/32 на горизонте 10 с для SOL |
| `label_horizons` | `[50, 100, 300]` | 5 с / 10 с / 30 с на сетке 100 мс |
| `oi_history_window_ms` | `3600000` | Лукбэк для `oi_change_pct_1h` |
| `liq_window_ms` | `60000` | Окно агрегатов ликвидаций |

### [`conf/model/default.yaml`](../conf/model/default.yaml)

`seq_len: 50` (5 с), `ob_depth: 50` (**должен совпадать с `ingestion.lob_depth`**),
`in_features_flow: 9`, `in_features_ctx: 17`, `d_model: 64`, `d_ctx: 64`,
`n_lob_blocks: 2`, `flow_encoder: gru`, `gru_layers: 2`, `n_classes: 3`,
`dropout: 0.1`, `use_ctx: true`.

### [`conf/train/default.yaml`](../conf/train/default.yaml)

`batch_size: 1024`, `max_epochs: 12`, `lr: 1e-3`, `weight_decay: 1e-4`,
`patience: 3`, `grad_clip_val: 1.0`, `use_focal_loss: false`, `focal_gamma: 2.0`,
`val_metric: val_f1_macro`, `seed: 42`, `window_stride: 10`, `num_workers: 4`.

### [`conf/signals/default.yaml`](../conf/signals/default.yaml)

`horizon_index: 1` (горизонт 10 с), `min_confidence: 0.55`, `allow_short: false`.

### [`conf/risk/default.yaml`](../conf/risk/default.yaml)

`max_position_notional: 100.0`, `max_leverage: 1.0`, `max_spread_bps: 5.0`,
`stale_data_ms: 1000`, `cooldown_after_trade_ms: 5000`, `min_confidence: 0.55`,
`allow_short: false`.

### [`conf/inference/default.yaml`](../conf/inference/default.yaml)

`checkpoint_path: null` и `normalizer_stats_path: null` — **обязательны**
при живом запуске; `device: auto`, `warmup_action: NO_TRADE`.

### [`conf/execution/default.yaml`](../conf/execution/default.yaml)

| Ключ | Значение | Смысл |
|---|---|---|
| `order_notional` | `100.0` | Размер позиции в USDT |
| `entry_order_type` | `PostOnly` | Пассивный вход ради мейкер-ребейта |
| `exit_order_type` | `PostOnly` | Сначала пассивно, потом пересечь |
| `order_timeout_ms` | `1000` | Отмена неисполненного входа |
| `exit_fallback_ms` | `2000` | Окно, после которого выход кроссится |
| `hold_snapshots` | `null` | → `features.label_horizons[signals.horizon_index]` = 100 → 10 с |
| `take_profit_bps` / `stop_loss_bps` | `5.0` / `5.0` | ≈1.4σ движения за 10 с на SOL |
| `qty_step` / `price_tick` / `min_order_qty` | `0.1` / `0.01` / `0.1` | Правила инструмента SOLUSDT |
| `demo` | `true` | Bybit **Demo** (не testnet): реальные данные, виртуальные деньги |
| `dry_run` | `false` | `true` → решать, но ничего не выставлять |

### [`conf/backtest/default.yaml`](../conf/backtest/default.yaml)

`split: test`, `checkpoint_path: null` (обязателен), `device: auto`,
`initial_equity: 10000.0`, `batch_size: 512`, `feed_dir: data/hft`,
`report_dir: data/reports`, `recorder_buffer_size: 10_000_000`,
комиссии/очередь/латентность — см. [12.4](#124-модели-очереди-и-латентности).

### [`conf/telemetry/default.yaml`](../conf/telemetry/default.yaml)

`rollup_interval_s: 60.0`, `window_capacity: 4096` (≈7 мин при 10 Гц).

### Переопределение из CLI

```bash
python -m bot.features.pipeline features.start_date=2025-08-07 features.end_date=2025-08-10
python -m bot.backtest.run backtest.split=val signals.min_confidence=0.6
```

---

## 15. Сквозной пример одной сделки

Проследим один снапшот от WS-кадра до реализованного PnL. Числа — из дефолтных
конфигов, цена SOL условная.

### Шаг 1 — кадр приходит

```
12:00:00.000  WS orderbook.50.SOLUSDT → OrderBookManager._data = {...}
12:00:00.043  WS publicTrade → Trade(qty=12.4, side="Buy") → accumulator (0.1 с бакет)
12:00:00.071  WS publicTrade → Trade(qty=3.1,  side="Sell")
12:00:00.100  таймер: OrderBookSnapshot(ts=1756...100, bids=[[182.34,410.2],...50],
                                        asks=[[182.35,388.7],...50], is_reset=False)
              ├→ StorageWriter.put_nowait → INSERT INTO snapshots
              └→ LiveTrader.on_snapshot   → очередь (maxsize=100)
```

### Шаг 2 — фичи

```
ob[0]   = [(182.34-182.34)/182.34, log1p(410.2), (182.35-182.35)/182.35, log1p(388.7)]
        = [0.0, 6.019, 0.0, 5.966]
ob[1..49] — уровни глубже, отступы растут
flow    = extract_flow_features([{Buy,12.4}, {Sell,3.1}])
        = [log1p(12.4), log1p(3.1), log1p(9.3), 2, 0.6, log1p(12.4), log1p(7.75), 1, 1]
        → normalize_only → z-вектор, например [1.42, -0.31, 1.10, 0.05, 1.88, ...]
ctx     = ctx_builder.snapshot(ts) → 17 значений (mark, index, basis, OI, funding, ...)

RingBuffer.append × 3 → буферы полны (50 снапшотов = 5 с истории)
```

### Шаг 3 — модель и сигнал

```
tensors: ob [1,50,50,4], flow [1,50,9], ctx [1,50,17]
logits  [1,3,3] → softmax → [[0.21,0.45,0.34],
                             [0.13,0.25,0.62],   ← horizon_index = 1 (10 с)
                             [0.30,0.40,0.30]]

argmax(горизонт 1) = 2 (UP), confidence = 0.62 >= min_confidence 0.55
→ TradingSignal(BUY, confidence=0.62, reason=None)
```

### Шаг 4 — риск-гейт

```
stale?        now - ts = 8 мс          < 1000     ✓
spread?       (182.35-182.34)/182.345*1e4 = 0.55  < 5.0 bps  ✓
confidence?   0.62 >= 0.55                        ✓
short?        действие BUY                        ✓
cooldown?     последний fill 42 с назад > 5 с     ✓
нотионал?     |0 * mid + 100| = 100 <= 100        ✓
плечо?        100 / 9987.4 = 0.010 <= 1.0         ✓
→ RiskDecision(approved=True, reason="approved") → InferenceStatus.APPROVED
```

### Шаг 5 — намерение и ордер

```
reference_price = best_bid = 182.34   (PostOnly BUY → пассивная сторона)
qty = floor(100 / 182.34, 0.1) = floor(0.5484…) = 0.5      → нотионал 91.17 USDT
0.5 >= min_order_qty 0.1  ✓

place_order_params →
  {"category":"linear","symbol":"SOLUSDT","side":"Buy","orderType":"Limit",
   "qty":"0.5","price":"182.34","timeInForce":"PostOnly","reduceOnly":False,
   "orderLinkId":"sp-e-b-1756...100"}
→ HTTP.place_order(**params)  через asyncio.to_thread
→ OrderEvent("submitted", latency_ms=31)
→ _RestingOrder(deadline_ms = now + 1000)
```

### Шаг 6 — исполнение

```
12:00:00.412  приватный WS order_stream: orderStatus="Filled", avgPrice="182.34"
              → AccountState._orders["sp-e-b-..."] = OrderUpdate(...)
12:00:00.500  следующий снапшот → _reconcile_resting видит is_filled
              → OrderEvent("filled", fill_price=182.34, latency_ms=400)
              → open_position():
                   TP = 182.34 * 1.0005 = 182.43117
                   SL = 182.34 * 0.9995 = 182.24883
                   hold_deadline = 12:00:00.412 + 10 000 мс = 12:00:10.412
```

Если бы ордер не исполнился, на снапшоте после `12:00:01.100` сработал бы
`deadline_ms` → `cancel_order` → `OrderEvent("cancelled")`, и петля вернулась бы
в состояние «свободен».

### Шаг 7 — удержание и выход

```
12:00:00.6 … 12:00:10.4 — на каждом снапшоте decide_exit():
    exit_price = best_bid (позиция BUY)
    SL пробит?  best_bid <= 182.24883 → нет
    TP взят?    best_bid >= 182.43117 → нет
    hold?       now < 12:00:10.412    → нет
    → record(HOLDING)

12:00:07.900  best_bid = 182.44 >= TP → ExitDecision(TAKE_PROFIT, 182.44)
              build_exit_intent(side=SELL, reduce_only=True, PostOnly, limit=best_ask)
              → OrderEvent("submitted"), _exit_pending = True
12:00:09.900  прошло 2000 мс = exit_fallback_ms → should_cross_exit → True
              → cancel + маркет-выход → OrderEvent("submitted")
12:00:10.100  account.is_flat → OrderEvent("filled", exit_reason="TAKE_PROFIT")
              позиция закрыта
```

### Шаг 8 — экономика этой сделки

| Компонент | bps | USDT (нотионал 91.17) |
|---|---|---|
| Движение цены до TP | +5.0 | +0.0456 |
| Мейкер-ребейт на входе (`maker_fee_rate = -0.0001`) | +1.0 | +0.0091 |
| **Выход мейкером** | +1.0 | +0.0091 |
| **Выход тейкером** (`taker_fee_rate = 0.00055`) | −5.5 | −0.0501 |

```
выход мейкером: +5.0 +1.0 +1.0 = +7.0 bps
выход тейкером: +5.0 +1.0 −5.5 = +0.5 bps
```

Отсюда вся конструкция пассивного выхода с эскалацией: тейкерская комиссия
(5.5 bps) **больше всего тейк-профита** (5 bps). Метрика `maker_share` в роллапе
— не косметика, а фактически знак итогового PnL.

### Шаг 9 — что осталось в базе

```sql
-- 100 строк за эти 10 секунд:
SELECT status, COUNT(*) FROM decisions WHERE timestamp_ms BETWEEN ... GROUP BY status;
-- APPROVED 1 | HOLDING 98 | WAITING 1

SELECT event, side, qty, fill_price, latency_ms, exit_reason FROM order_events
WHERE order_link_id LIKE 'sp-%';
-- submitted Buy  0.5 NULL   31  NULL
-- filled    Buy  0.5 182.34 400 NULL
-- submitted Sell 0.5 NULL   28  NULL
-- filled    Sell 0.5 182.44 200 TAKE_PROFIT
```

---

## 16. Runbook — команды end-to-end

> Тесты и скрипты запускаются из `.venv` проекта: `.venv\Scripts\python -m ...`.
> Точки входа с `uvloop` ([`bot/main.py`](../bot/main.py) и
> [`bot/execution/run.py`](../bot/execution/run.py)) **не запускаются на Windows** —
> `uvloop` не поддерживает эту платформу. Для live-приёма и торговли нужен
> Linux/WSL.

### 0. Секреты

```bash
# .env в корне проекта, в git не попадает
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
```

### 1. Исторические данные → SQLite + raw-кэш

```bash
python -m bot.data.historical_loader \
  historical.start_date=2025-08-07 historical.end_date=2025-08-20
```
Результат: `data/db/SOLUSDT_2025-08-*.db` (снапшоты + сделки) и
`data/raw/*.zip`, `*.csv.gz`. Идемпотентно: существующие дни пропускаются.

### 2. Фичи и метки → .npy

```bash
python -m bot.features.pipeline \
  features.start_date=2025-08-07 features.end_date=2025-08-20
```
Результат: `data/features/SOLUSDT/` (см. [7.7](#77-выход-стадии-c)).
Занимает часы и ~10 ГБ на две недели — memmap пишет насквозь, память не растёт.

### 3. Обучение → checkpoint

```bash
python -m bot.training.train
# ablation без контекста:
python -m bot.training.train model.use_ctx=false
```
Результат: `lightning_logs/version_{k}/checkpoints/best-*.ckpt`.

### 4. Бэктест на hftbacktest

```bash
python -m bot.backtest.run \
  backtest.checkpoint_path=lightning_logs/version_0/checkpoints/best-....ckpt \
  backtest.split=test
```
Первый запуск дополнительно конвертирует архивы в `data/hft/` (долго).
Результат: `data/reports/SOLUSDT_test/{record.npz,run.json,report.html}`.

Перерисовать отчёт без реплея:
```bash
python -m bot.backtest.report data/reports/SOLUSDT_test
```

### 5. Только приём live-данных (без торговли, Linux/WSL)

```bash
python -m bot.main
```

### 6. Живая торговля — сначала всухую

```bash
python -m bot.execution.run \
  inference.checkpoint_path=lightning_logs/version_0/checkpoints/best-....ckpt \
  inference.normalizer_stats_path=data/features/SOLUSDT/normalizer_stats.npz \
  execution.dry_run=true execution.demo=true
```
`dry_run=true` — решения принимаются и пишутся в телеметрию, ордера не
выставляются. Это правильный первый прогон: он даёт `loop_ms`, `book_age_ms`,
`inference_us` и, главное, `flow_skew` — прежде чем на кону окажутся деньги.

Снять `dry_run` после того, как роллап показал вменяемые числа:
```bash
python -m bot.execution.run inference.checkpoint_path=... \
  inference.normalizer_stats_path=... execution.demo=true
```

### 7. Оценка живой сессии

```python
from pathlib import Path
from bot.telemetry.evaluation import load_decisions, load_mid_prices, score

paths = [Path("data/db/SOLUSDT_2026-08-29.db")]
ts, mids, resets = load_mid_prices(paths)
result = score(load_decisions(paths), ts, mids, resets, horizon=100, alpha=0.0001)
print(result.f1_macro, result.hit_rate_by_confidence)
```
`f1_macro` здесь напрямую сравнима с `val_f1_macro` чекпоинта.

### 8. Тесты

```bash
.venv\Scripts\python -m pytest              # всё
.venv\Scripts\python -m pytest tests/features -q
```

---

## 17. Тесты — что чем покрыто

Все тесты синтетические: без сети, без живых файлов, in-memory SQLite для
пайплайна.

| Каталог | Файлы | Что проверяется |
|---|---|---|
| [`tests/data/`](../tests/data/) | [`test_orderbook.py`](../tests/data/test_orderbook.py), [`test_storage.py`](../tests/data/test_storage.py), [`test_context_poller.py`](../tests/data/test_context_poller.py), [`test_historical_loader.py`](../tests/data/test_historical_loader.py), [`test_raw_cache.py`](../tests/data/test_raw_cache.py) | Детект реконнекта, ротация БД и backpressure, дедуп поллера, восстановление книги из дельт, кэш архивов |
| [`tests/features/`](../tests/features/) | [`test_ob_serializer.py`](../tests/features/test_ob_serializer.py), [`test_flow_features.py`](../tests/features/test_flow_features.py), [`test_context_features.py`](../tests/features/test_context_features.py), [`test_labels.py`](../tests/features/test_labels.py), [`test_normalizer.py`](../tests/features/test_normalizer.py), [`test_pipeline.py`](../tests/features/test_pipeline.py) | Кодировки фичей, окна контекста, формула меток и `valid_mask`, save/load нормализатора, сквозной прогон пайплайна |
| [`tests/training/`](../tests/training/) | [`test_encoders.py`](../tests/training/test_encoders.py), [`test_fusion.py`](../tests/training/test_fusion.py), [`test_hybrid.py`](../tests/training/test_hybrid.py), [`test_dataset.py`](../tests/training/test_dataset.py), [`test_losses.py`](../tests/training/test_losses.py), [`test_metrics.py`](../tests/training/test_metrics.py), [`test_train.py`](../tests/training/test_train.py) | Формы тензоров на каждом уровне, оконный датасет и `stride`, маскирование flat в лоссе, F1 |
| [`tests/signals/`](../tests/signals/) | [`test_policy.py`](../tests/signals/test_policy.py) | Все ветви `generate_signal` и их `reason` |
| [`tests/risk/`](../tests/risk/) | [`test_gate.py`](../tests/risk/test_gate.py) | Все девять отказов и их порядок |
| [`tests/inference/`](../tests/inference/) | [`test_buffer.py`](../tests/inference/test_buffer.py), [`test_adapters.py`](../tests/inference/test_adapters.py), [`test_preprocessor.py`](../tests/inference/test_preprocessor.py), [`test_predictor.py`](../tests/inference/test_predictor.py) | Кольцевой буфер, соответствие адаптеров формам Фазы 2, `is_ready`, загрузка чекпоинта |
| [`tests/execution/`](../tests/execution/) | [`test_lifecycle.py`](../tests/execution/test_lifecycle.py), [`test_orders.py`](../tests/execution/test_orders.py), [`test_account.py`](../tests/execution/test_account.py), [`test_live.py`](../tests/execution/test_live.py) | Сайзинг, приоритет выходов, форматирование kwargs, парсинг приватного WS, автомат `LiveTrader` |
| [`tests/backtest/`](../tests/backtest/) | [`test_prepare.py`](../tests/backtest/test_prepare.py), [`test_strategy.py`](../tests/backtest/test_strategy.py), [`test_run.py`](../tests/backtest/test_run.py), [`test_report.py`](../tests/backtest/test_report.py), [`test_integration.py`](../tests/backtest/test_integration.py) | Подготовка фида, цикл стратегии на фейковом `hbt` ([`conftest.py`](../tests/backtest/conftest.py)), выравнивание расписания, отчёт |
| [`tests/telemetry/`](../tests/telemetry/) | [`test_collector.py`](../tests/telemetry/test_collector.py), [`test_evaluation.py`](../tests/telemetry/test_evaluation.py), [`test_storage_roundtrip.py`](../tests/telemetry/test_storage_roundtrip.py) | Счётчики и квантили, скоринг live-предсказаний, round-trip `DecisionRecord`/`OrderEvent` через SQLite |

---

## 18. Инварианты, подводные камни и расхождения

Список того, что легко упустить, читая код по кускам. Пункты 1–4 —
содержательные находки, а не косметика.

### 1. `CLAUDE.md` местами отстал от кода

| Утверждение в [`CLAUDE.md`](../CLAUDE.md) | Фактически в `conf/` |
|---|---|
| Символ `BTCUSDT` | `SOLUSDT` |
| `label_horizons` по умолчанию `[1,5,10]` | `[50, 100, 300]` (5/10/30 с) |
| `label_alpha = 0.001` | `0.0001` |
| «market exit (protective exits must fill)» | `exit_order_type: PostOnly` + `exit_fallback_ms` |
| Файлы Фазы 2 без `valid_mask.npy` | `valid_mask.npy` пишется |

Правильный источник истины — YAML и код.

### 2. На исторических данных контекст — константа

Архивы Bybit несут только книгу и сделки, поэтому в БД, созданных
[`historical_loader`](../bot/data/historical_loader.py), нет таблиц
`ticker_context`, `liquidations`, `long_short_ratio`.
[`_query_all`](../bot/features/replay.py#L36) на отсутствующую таблицу возвращает
пустой список — это осознанное поведение, а не ошибка.

Следствие: `ContextBuilder` никогда не обновляется, и все 17 контекстных фичей
принимают своё дефолтное значение на **каждой** строке:

```
ctx = [0,0,0,0,0,0, 1.0, 0,0,0,0,0,0,0,0, 1.0, 0]
       ↑ funding_sign = 1.0 (0 >= 0)      ↑ ls_ratio = 1.0 (sell_ratio <= 0)
```

Это константа. `CtxEncoder` при `use_ctx=true` вносит в таком случае только
смещение (bias), а `GatedFusion` — постоянный вклад в гейт. То есть на данных из
архивов ветка контекста фактически ничего не даёт, хотя занимает параметры и
компьют. Варианты: обучать с `model.use_ctx=false`, либо строить фичи по
дням, собранным live-приёмом (там ticker/liq/ls есть).

### 3. Экономика: тейкерский выход дороже всего тейк-профита

`take_profit_bps = 5.0`, `taker_fee_rate = 0.00055` = 5.5 bps. Разбор в
[разделе 15, шаг 8](#шаг-8--экономика-этой-сделки): при выходе тейкером
теоретический край сделки ≈ +0.5 bps, при выходе мейкером ≈ +7 bps. Поэтому:

- `exit_order_type: PostOnly` с эскалацией — не оптимизация, а условие
  прибыльности;
- `maker_share` в роллапе и `slippage_bps` — метрики первого порядка;
- `exit_fallback_ms = 2000` при `hold_ms = 10000` означает, что пассивная
  попытка занимает пятую часть удержания.

### 4. Вход в бэктесте всегда PostOnly

[`strategy.py:220-222`](../bot/backtest/strategy.py#L220-L222) вызывает
`build_entry_intent(..., OrderType.POST_ONLY, spec)` жёстко, тогда как
`LiveTrader` использует `cfg.execution.entry_order_type`. При значении по
умолчанию (`PostOnly`) они совпадают; если поставить `entry_order_type: Market`,
бэктест и live разойдутся молча.

### 5. «Замороженная статистика трейна» — это последние 500 строк

[`RollingNormalizer._recompute_stats`](../bot/features/normalizer.py#L51) считает
mean/std по deque'у длиной `normalizer_window = 500`. Пайплайн прогоняет весь
трейн через `update_and_normalize`, поэтому в `normalizer_stats.npz` сохраняется
статистика **последних 500 строк трейна** — примерно 50 секунд рынка на границе
train/val, а не всего трейн-сплита.

Онлайн-путь и `FlowSkew` меряют дрейф относительно именно этой статистики.
Если те 50 секунд были нетипичными, `z ~ N(0,1)` не выполнится даже на здоровом
потоке. Это не мешает обучению (трейн z-scored'ился скользящим окном), но
делает live-нормализацию чувствительной к одной случайной точке во времени.

### 6. Flow-бакет в live шире, чем в обучении, при просадке петли

См. [9.3](#93-разница-онлайна-и-оффлайна-в-flow-бакете). Диагностируется
`loop_ms` p99 и `flow_skew`.

### 7. Выходы не джойнятся по `order_link_id`

При закрытии позиции [`_manage_position`](../bot/execution/live.py#L208) пишет
`OrderEvent` с `link_id=f"exit-{self._exit_submitted_ms}"`, а не с реальным
`orderLinkId`, под которым ордер отправлялся. Событие `submitted` для этого же
выхода несёт настоящий id. Значит join `submitted → filled` по `order_link_id`
для выходов не сойдётся (для входов сойдётся — там используется
`self._resting.link_id`).

### 8. `valid_mask.npy` никем не читается

Пишется в [`pipeline.py:202`](../bot/features/pipeline.py#L202), но ни обучение,
ни бэктест его не грузят: его смысл уже вшит в `labels`/`flat_mask` строками
выше. Файл диагностический.

### 9. Одна позиция за раз — по построению

И `LiveTrader`, и `run_strategy` держат ровно одну позицию. Поэтому риск-гейт в
бэктесте всегда получает `PositionState(quantity=0)`
([`strategy.py:206-210`](../bot/backtest/strategy.py#L206-L210)), а проверки
нотионала и плеча де-факто сводятся к проверке одного ордера. `AccountState`
при этом умеет знаковую позицию — на будущее.

### 10. `risk.max_position_notional == execution.order_notional == 100.0`

Порог равен размеру ордера, то есть проверка `>` проходит ровно на границе.
Любое увеличение `order_notional` без правки `risk` даст
`max_position_notional_exceeded` на каждом сигнале — и это будет видно как
доминирующая причина в `risk_rejected`.

### 11. Согласованность K и горизонтов

- `cfg.model.ob_depth` **обязан** равняться `cfg.ingestion.lob_depth` — иначе
  форма `ob_raw` не сойдётся с ожиданием `LOBEncoder`.
- `n_horizons` нигде не хранится в конфиге модели: он всегда выводится как
  `len(cfg.features.label_horizons)` — в
  [`train.py:102`](../bot/training/train.py#L102),
  [`run.py:84`](../bot/execution/run.py#L84) и
  [`backtest/run.py:221`](../bot/backtest/run.py#L221). Смена списка горизонтов
  делает старые чекпоинты несовместимыми (изменится размер `head`).
- `cfg.signals.horizon_index` должен быть валидным индексом в этом списке;
  [`generate_signal`](../bot/signals/policy.py#L28) бросает `ValueError`, если нет.

### 12. Таймстемпы

- Снапшоты в live помечаются **локальным** временем машины, а не биржевым.
- Исторические снапшоты помечаются полем `ts` из архива (биржевое время).
- `book_age_ms = now_ms - snapshot.timestamp_ms` в live измеряет задержку петли,
  а не сети.
- Ротация БД идёт по UTC-дате таймстемпа **записи**, а не по времени открытия
  файла.

### 13. Правила из [`CLAUDE.md`](../CLAUDE.md), которые держит код

- Никаких прокси/обёрток вокруг pybit.
- `try/except` только на известное исключение с реальным восстановлением. Во
  всём `bot/` их семь: `asyncio.TimeoutError` (поллер и цикл `StorageWriter`),
  `asyncio.QueueEmpty` ×2 (drop-oldest и `_flush`), `StopIteration` и
  `ValueError` (детект заголовка CSV), `ImportError` для Mamba с подсказкой по
  установке. Ни одного «на всякий случай».
- Логирование только в точках входа и в жизненном цикле ордеров `LiveTrader` —
  ни одного логгера в экстракторах, энкодерах и чистых функциях сигнала/риска.
- `Decimal` в памяти → `str` в БД → `float32` на границе фичей.

---

## 19. Текущее состояние данных на диске

На момент написания документа:

| Путь | Состояние |
|---|---|
| `data/db/` | 14 дневных файлов `SOLUSDT_2025-08-07 … 2025-08-20`, ~861 тыс. снапшотов и ~780 тыс. сделок в день. Только таблицы `snapshots` и `trades` (загружено историческим загрузчиком) |
| `data/raw/` | Кэшированные архивы за тот же диапазон — готовы для `backtest/prepare.py` |
| `data/features/SOLUSDT/` | `ob_raw.npy` (5.5 ГБ), `flow_features.npy`, `ctx_features.npy`, `top_of_book.npy` — все на `N = 6 903 304` строки, то есть на первые **8 дней** (`08-07 … 08-14`) из 14; полный диапазон дал бы `N = 12 070 197` |
| `data/hft/`, `data/reports/` | Отсутствуют — бэктест ещё не запускался |
| `lightning_logs/` | Отсутствует — обучение ещё не запускалось |

> ⚠️ **Прогон пайплайна фичей не завершён.** В `data/features/SOLUSDT/` есть
> только четыре memmap-массива, которые пишутся в главном цикле, но нет
> `labels.npy`, `flat_mask.npy`, `valid_mask.npy`, `timestamps_ms.npy`,
> `is_reset.npy` и `normalizer_stats.npz` — они сохраняются в самом конце
> [`run_pipeline`](../bot/features/pipeline.py#L196-L205). Кроме того,
> `flow_features.npy` в этом состоянии **не нормализован**: z-score применяется
> на том же финальном этапе.
>
> До повторного полного прогона [`bot/features/pipeline.py`](../bot/features/pipeline.py)
> ни обучение, ни бэктест запустить нельзя — оба грузят `labels.npy`.
> Прогон надо повторить целиком: `N` фиксируется в начале по
> `count_snapshots`, дозаписать недостающие дни в существующие memmap нельзя.

### Что дальше по плану проекта

1. Дозапустить пайплайн фичей на полном диапазоне.
2. Обучить модель; решить вопрос с константным контекстом ([пункт 2](#2-на-исторических-данных-контекст--константа)) —
   либо `model.use_ctx=false`, либо перейти на дни, собранные live-приёмом.
3. Прогнать бэктест на `split=test`, прочитать воронку решений в отчёте.
4. Сессия на Demo с `execution.dry_run=true` — снять реальные `loop_ms`,
   `time_to_fill_ms` и `flow_skew`.
5. Заменить синтетическую латентность бэктеста измеренной
   ([12.4](#124-модели-очереди-и-латентности)) и перепрогнать.
