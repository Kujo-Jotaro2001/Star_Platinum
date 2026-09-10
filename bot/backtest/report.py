"""Render a backtest run as one self-contained HTML file.

hftbacktest's own `Stats.plot()` puts price on a twin axis against equity; two
different scales on one chart cannot be compared by eye, so the series here are
all indexed to percent of book size and share a single axis. The decision funnel
has no equivalent in hftbacktest at all — those counters belong to our strategy
and are what separates "the model was wrong" from "the orders never filled".
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from bot.telemetry.types import PipelineCounters

_DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{head}</head>
<body>
{body}</body>
</html>
"""

RECORD_FILE = "record.npz"
RUN_FILE = "run.json"
REPORT_FILE = "report.html"

CONTRACT_SIZE = 1.0
MAX_POINTS = 1500


@dataclass(frozen=True, slots=True)
class RunMeta:
    symbol: str
    split: str
    start: str
    end: str
    checkpoint: str
    queue_model: str
    latency_model: str
    maker_fee_rate: float
    taker_fee_rate: float
    hold_ms: int
    take_profit_bps: float
    stop_loss_bps: float
    order_notional: float
    initial_equity: float
    min_confidence: float
    horizon_index: int
    feed_latency_ns: int
    latency_is_measured: bool


@dataclass
class ReportData:
    meta: dict
    metrics: list[dict]
    headline: dict
    series: dict
    funnel: list[dict]
    exits: list[dict]
    warnings: list[str] = field(default_factory=list)


def equity_curves(record: np.ndarray, initial_equity: float) -> dict[str, np.ndarray]:
    """Equity with and without fees, plus the buy-and-hold benchmark.

    hftbacktest's balance is trading cash flow and opens at zero, so the account
    value is the configured book size plus that flow.
    """
    price = _filled_prices(record["price"].astype(np.float64))
    equity_wo_fee = (
        record["balance"].astype(np.float64)
        + record["position"].astype(np.float64) * price * CONTRACT_SIZE
    )
    equity = equity_wo_fee - record["fee"].astype(np.float64)

    account = initial_equity + equity
    peak = np.maximum.accumulate(account)
    drawdown = np.where(peak > 0, (peak - account) / peak * 100.0, 0.0)

    return {
        "net_pct": equity / initial_equity * 100.0,
        "gross_pct": equity_wo_fee / initial_equity * 100.0,
        "price_pct": (price / price[0] - 1.0) * 100.0 if price[0] else np.zeros_like(price),
        "position": record["position"].astype(np.float64),
        "drawdown_pct": drawdown,
    }


def _filled_prices(price: np.ndarray) -> np.ndarray:
    """Carry the last known price across steps where the book had no top.

    A feed opens before its first snapshot is applied, so the first few steps
    have no best bid or ask and hftbacktest reports NaN. Position is flat there,
    so nothing is misstated by carrying the first real price backwards, and the
    alternative is a chart series that cannot be serialised at all.
    """
    finite = np.isfinite(price)
    if finite.all():
        return price
    if not finite.any():
        return np.zeros_like(price)

    idx = np.where(finite, np.arange(len(price)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = price[idx]
    # leading gap: nothing to carry forward from, so carry the first price back
    first = int(np.argmax(finite))
    filled[:first] = price[first]
    return filled


def downsample(
    timestamps_ns: np.ndarray,
    curves: dict[str, np.ndarray],
    max_points: int = MAX_POINTS,
) -> dict:
    """Thin the series for the browser, keeping the worst drawdown in each bucket.

    A plain stride would drop the spikes that matter most on the underwater plot.
    """
    n = len(timestamps_ns)
    if n <= max_points:
        return {
            "t": (timestamps_ns // 1_000_000).tolist(),
            **{name: _round(values) for name, values in curves.items()},
        }

    edges = np.linspace(0, n, max_points + 1, dtype=np.int64)
    starts, ends = edges[:-1], edges[1:]
    keep = ends - 1

    out = {"t": (timestamps_ns[keep] // 1_000_000).tolist()}
    for name, values in curves.items():
        if name == "drawdown_pct":
            out[name] = _round(
                np.array([values[a:b].max() for a, b in zip(starts, ends)])
            )
        else:
            out[name] = _round(values[keep])
    return out


def build_report_data(
    record: np.ndarray,
    summary: dict[str, float | str],
    strategy: PipelineCounters,
    meta: RunMeta,
) -> ReportData:
    curves = equity_curves(record, meta.initial_equity)
    series = downsample(record["timestamp"].astype(np.int64), curves)

    net_return = float(curves["net_pct"][-1])
    gross_return = float(curves["gross_pct"][-1])
    benchmark = float(curves["price_pct"][-1])
    max_dd = float(curves["drawdown_pct"].max())

    fill_rate = (
        strategy.orders_filled / strategy.orders_submitted
        if strategy.orders_submitted
        else 0.0
    )
    approval_rate = (
        strategy.signals_approved / strategy.signals_actionable
        if strategy.signals_actionable
        else 0.0
    )

    warnings: list[str] = []
    if not meta.latency_is_measured:
        warnings.append(
            f"Order latency is derived from an assumed feed latency of "
            f"{meta.feed_latency_ns / 1e6:.0f} ms. Bybit's public history carries no "
            f"local timestamp, so nothing here measures real latency — read the fill "
            f"rate as a modelling output, not an observation."
        )
    if strategy.orders_submitted == 0:
        warnings.append(
            "No orders were submitted. Every signal was filtered out before it "
            "reached the book — read the funnel below to see where."
        )

    return ReportData(
        meta=asdict(meta),
        headline={
            "net_return": net_return,
            "gross_return": gross_return,
            "fee_drag": gross_return - net_return,
            "benchmark": benchmark,
        },
        metrics=[
            {"label": "Net return", "value": _pct(net_return), "tone": _tone(net_return)},
            {"label": "Sharpe", "value": _num(summary.get("SR"))},
            {"label": "Sortino", "value": _num(summary.get("Sortino"))},
            {"label": "Max drawdown", "value": _pct(-max_dd), "tone": "critical" if max_dd > 0 else None},
            {"label": "Return / MDD", "value": _num(summary.get("ReturnOverMDD"))},
            {"label": "Trades per day", "value": _num(summary.get("DailyNumberOfTrades"), 1)},
        ],
        series=series,
        funnel=[
            {"label": "Signals actionable", "value": strategy.signals_actionable,
             "note": "model chose a direction above the confidence floor"},
            {"label": "Approved by risk", "value": strategy.signals_approved,
             "note": f"{approval_rate:.0%} of actionable"},
            {"label": "Orders submitted", "value": strategy.orders_submitted,
             "note": "entries and exits"},
            {"label": "Filled", "value": strategy.orders_filled,
             "note": f"{fill_rate:.0%} of submitted"},
            {"label": "Cancelled on timeout", "value": strategy.orders_cancelled,
             "note": "post-only never traded through"},
            {"label": "Expired or rejected", "value": strategy.orders_expired,
             "note": "post-only would have crossed"},
        ],
        exits=[
            {"label": _exit_label(reason), "value": count}
            for reason, count in sorted(
                strategy.exits_by_reason.items(), key=lambda kv: -kv[1]
            )
        ],
        warnings=warnings,
    )


def render_html(data: ReportData, standalone: bool = True) -> str:
    """The report as HTML.

    `standalone=True` gives a complete document to open from disk. The fragment
    form exists because some hosts supply their own document shell and wrapping
    a second one inside it breaks the page.
    """
    body = _BODY.replace("__DATA__", json.dumps(asdict(data), allow_nan=False))
    if not standalone:
        return _HEAD + body
    return _DOCUMENT.format(head=_HEAD, body=body)


def write_report(path: Path, data: ReportData) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(data), encoding="utf-8")
    return path


def save_run(
    run_dir: Path,
    record: np.ndarray,
    summary: dict,
    strategy: PipelineCounters,
    meta: RunMeta,
) -> Path:
    """Persist everything the report needs, so it can be redrawn without replaying."""
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / RECORD_FILE, record=record)
    (run_dir / RUN_FILE).write_text(
        json.dumps(
            {"summary": summary, "strategy": asdict(strategy), "meta": asdict(meta)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return write_report(
        run_dir / REPORT_FILE, build_report_data(record, summary, strategy, meta)
    )


def load_run(run_dir: Path) -> tuple[np.ndarray, dict, PipelineCounters, RunMeta]:
    with np.load(run_dir / RECORD_FILE) as npz:
        record = npz["record"]
    saved = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    return (
        record,
        saved["summary"],
        PipelineCounters(**saved["strategy"]),
        RunMeta(**saved["meta"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw a saved backtest run as report.html."
    )
    parser.add_argument("run_dir", type=Path, help="directory written by bot.backtest.run")
    args = parser.parse_args()

    record, summary, strategy, meta = load_run(args.run_dir)
    path = write_report(
        args.run_dir / REPORT_FILE,
        build_report_data(record, summary, strategy, meta),
    )
    print(path)


def _exit_label(reason: str) -> str:
    return reason.replace("_", " ").capitalize()


def _round(values: np.ndarray) -> list[float]:
    return [round(float(v), 6) for v in values]


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def _num(value: float | str | None, digits: int = 2) -> str:
    if value is None or isinstance(value, str):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def _tone(value: float) -> str:
    return "good" if value > 0 else "critical" if value < 0 else "neutral"


_HEAD = r"""<title>Backtest Debrief</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --plane:#F4F5F7; --surface:#FBFBFC; --ink:#12151A; --ink-2:#5A626E; --ink-3:#8B93A0;
  --grid:#E3E5EA; --rule:#D7DAE0; --axis:#C3C7CF;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219;
  --ring:rgba(18,21,26,.10);
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --plane:#0E1013; --surface:#16181C; --ink:#EEF1F5; --ink-2:#A3ACB9; --ink-3:#7C8593;
  --grid:#262A31; --rule:#2E333B; --axis:#3A4048;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --ring:rgba(238,241,245,.12);
}}
:root[data-theme="dark"]{
  --plane:#0E1013; --surface:#16181C; --ink:#EEF1F5; --ink-2:#A3ACB9; --ink-3:#7C8593;
  --grid:#262A31; --rule:#2E333B; --axis:#3A4048;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --ring:rgba(238,241,245,.12);
}
*{box-sizing:border-box}
html{color-scheme:light dark}
body{margin:0;background:var(--plane);color:var(--ink);font-family:var(--sans);line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 72px;display:flex;flex-direction:column;gap:28px}
.eyebrow{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-3)}
h1{font-family:var(--cond);font-weight:700;font-size:clamp(30px,4.5vw,44px);line-height:1.05;
  letter-spacing:-.01em;margin:6px 0 0;text-wrap:balance}
h2{font-family:var(--cond);font-weight:600;font-size:19px;letter-spacing:.005em;margin:0}
.sub{color:var(--ink-2);font-size:14px;margin:8px 0 0;max-width:64ch}

.masthead{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:32px;align-items:end;
  border-bottom:1px solid var(--rule);padding-bottom:24px}
.verdict{text-align:right}
.verdict .figure{font-family:var(--mono);font-weight:600;font-size:clamp(34px,6vw,52px);
  line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.verdict .cap{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-top:8px}
.good{color:var(--good)} .critical{color:var(--critical)} .neutral{color:var(--ink-2)}

.runbar{display:flex;flex-wrap:wrap;gap:0 28px;font-family:var(--mono);font-size:12px;color:var(--ink-2)}
.runbar b{font-weight:500;color:var(--ink-3)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.tile{background:var(--surface);padding:16px 18px}
.tile .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.tile .v{font-family:var(--mono);font-weight:600;font-size:22px;margin-top:7px;
  font-variant-numeric:tabular-nums;letter-spacing:-.01em}

.card{background:var(--surface);border:1px solid var(--ring);border-radius:3px;padding:20px 22px 16px}
.card header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:4px}
.card .hint{color:var(--ink-3);font-size:13px;margin:0 0 14px;max-width:70ch}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:820px){.two{grid-template-columns:1fr}.masthead{grid-template-columns:1fr}
  .verdict{text-align:left}}

.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.swatch{width:11px;height:11px;border-radius:2px;flex:none}

.plot{width:100%;display:block;overflow:visible}
.plot .gridline{stroke:var(--grid);stroke-width:1}
.plot .zero{stroke:var(--axis);stroke-width:1}
.plot .tick{font-family:var(--mono);font-size:10px;fill:var(--ink-3)}
.plot .endlabel{font-family:var(--mono);font-size:11px;font-weight:600}
.plot path.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.plot .crosshair{stroke:var(--axis);stroke-width:1;stroke-dasharray:3 3}
.plot .dot{stroke:var(--surface);stroke-width:2}

.tip{position:fixed;pointer-events:none;z-index:9;background:var(--surface);color:var(--ink);
  border:1px solid var(--ring);border-radius:3px;padding:9px 11px;font-family:var(--mono);
  font-size:11.5px;line-height:1.65;box-shadow:0 6px 24px rgba(0,0,0,.14);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.tip .h{color:var(--ink-3);margin-bottom:4px}
.tip i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}

.bars{display:flex;flex-direction:column;gap:9px;margin-top:4px}
.bar{display:grid;grid-template-columns:minmax(120px,190px) 1fr auto;gap:14px;align-items:center}
.bar .lab{font-size:13px}
.bar .lab small{display:block;color:var(--ink-3);font-size:11.5px;font-family:var(--mono)}
.bar .track{height:16px;background:var(--grid);border-radius:2px;overflow:hidden}
.bar .fill{height:100%;border-radius:2px;min-width:2px}
.bar .n{font-family:var(--mono);font-weight:600;font-size:14px;font-variant-numeric:tabular-nums;
  min-width:5ch;text-align:right}

table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid)}
th{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500}
td.n{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}

.note{border-left:2px solid var(--warning);background:var(--surface);border-radius:0 3px 3px 0;
  padding:14px 18px;font-size:13.5px;color:var(--ink-2)}
.note b{color:var(--ink);font-weight:600}
.toggle{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  background:none;border:1px solid var(--ring);color:var(--ink-2);border-radius:3px;
  padding:5px 10px;cursor:pointer}
.toggle:hover{color:var(--ink);border-color:var(--axis)}
.toggle:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
[hidden]{display:none!important}
@media (prefers-reduced-motion:no-preference){.plot path.line{transition:opacity .15s}}
</style>
"""

_BODY = r"""<div class="wrap" id="app"></div>
<div class="tip" id="tip" hidden></div>

<script>
const D = __DATA__;
const S = ['--s1','--s2','--s3'].map(v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim());
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtPct = v => (v>=0?'+':'') + v.toFixed(2) + '%';
const fmtTime = ms => new Date(ms).toISOString().slice(0,19).replace('T',' ');
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

const SERIES = [
  {key:'net_pct',   name:'Equity, net of fees'},
  {key:'gross_pct', name:'Equity, before fees'},
  {key:'price_pct', name:'Price (buy & hold)'},
];

function el(html){const t=document.createElement('template');t.innerHTML=html.trim();return t.content.firstChild;}

/* ---------- multi-series line chart, one axis, all series in % ---------- */
function lineChart(host, series, opts){
  const W=host.clientWidth||860, H=opts.height||280,
        M={t:14,r:opts.right??86,b:26,l:52};
  const t=D.series.t, n=t.length;
  const all=series.flatMap(s=>s.data);
  let lo=Math.min(...all), hi=Math.max(...all);
  if(lo===hi){lo-=1;hi+=1;}
  const pad=(hi-lo)*0.12; lo-=pad; hi+=pad;
  if(opts.zeroFloor && lo>0) lo=0;

  const X=i=>M.l+(n<2?0:i/(n-1))*(W-M.l-M.r);
  const Y=v=>M.t+(1-(v-lo)/(hi-lo))*(H-M.t-M.b);

  const ticks=5, parts=[];
  for(let k=0;k<=ticks;k++){
    const v=lo+(hi-lo)*k/ticks, y=Y(v);
    parts.push(`<line class="${Math.abs(v)<1e-9?'zero':'gridline'}" x1="${M.l}" y1="${y}" x2="${W-M.r}" y2="${y}"/>`);
    parts.push(`<text class="tick" x="${M.l-8}" y="${y+3.5}" text-anchor="end">${v.toFixed(opts.digits??1)}${opts.unit??'%'}</text>`);
  }
  const nt=Math.min(4,n-1);
  for(let k=0;k<=nt;k++){
    const i=Math.round(k/nt*(n-1));
    parts.push(`<text class="tick" x="${X(i)}" y="${H-8}" text-anchor="${k===0?'start':k===nt?'end':'middle'}">${fmtTime(t[i]).slice(11)}</text>`);
  }
  series.forEach((s,si)=>{
    const d=s.data.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
    parts.push(`<path class="line" d="${d}" stroke="${s.color}"/>`);
    if(opts.right!==0){
      const last=s.data[n-1];
      parts.push(`<text class="endlabel" x="${W-M.r+8}" y="${Y(last)+3.5}" fill="${s.color}">${fmtPct(last)}</text>`);
    }
  });
  parts.push(`<g id="ch-${opts.id}" style="display:none"><line class="crosshair" y1="${M.t}" y2="${H-M.b}"/>${
    series.map(s=>`<circle class="dot" r="4" fill="${s.color}"/>`).join('')}</g>`);
  parts.push(`<rect x="${M.l}" y="${M.t}" width="${W-M.l-M.r}" height="${H-M.t-M.b}" fill="transparent" data-hit="1"/>`);

  host.innerHTML=`<svg class="plot" viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(opts.aria)}">${parts.join('')}</svg>`;

  const svg=host.querySelector('svg'), g=svg.querySelector(`#ch-${opts.id}`), tip=document.getElementById('tip');
  svg.querySelector('[data-hit]').addEventListener('pointermove',e=>{
    const r=svg.getBoundingClientRect(), sx=(e.clientX-r.left)/r.width*W;
    const i=Math.max(0,Math.min(n-1,Math.round((sx-M.l)/(W-M.l-M.r)*(n-1))));
    g.style.display='';
    g.querySelector('line').setAttribute('x1',X(i)); g.querySelector('line').setAttribute('x2',X(i));
    g.querySelectorAll('circle').forEach((c,si)=>{c.setAttribute('cx',X(i));c.setAttribute('cy',Y(series[si].data[i]));});
    tip.hidden=false;
    tip.innerHTML=`<div class="h">${fmtTime(t[i])}</div>`+series.map(s=>
      `<div><i style="background:${s.color}"></i>${esc(s.name)} <b>${fmtPct(s.data[i])}</b></div>`).join('');
    const tw=tip.offsetWidth;
    tip.style.left=Math.min(e.clientX+14, innerWidth-tw-12)+'px';
    tip.style.top=(e.clientY+16)+'px';
  });
  svg.addEventListener('pointerleave',()=>{g.style.display='none';tip.hidden=true;});
}

/* ---------- horizontal bars ---------- */
function bars(host, rows, color, ordinal){
  const max=Math.max(1,...rows.map(r=>r.value));
  host.innerHTML=`<div class="bars">${rows.map((r,i)=>{
    const c = ordinal ? ordinal[Math.min(i,ordinal.length-1)] : color;
    return `<div class="bar"><div class="lab">${esc(r.label)}${r.note?`<small>${esc(r.note)}</small>`:''}</div>
      <div class="track"><div class="fill" style="width:${(r.value/max*100).toFixed(1)}%;background:${c}"></div></div>
      <div class="n">${r.value.toLocaleString()}</div></div>`;
  }).join('')}</div>`;
}

/* ---------- page ---------- */
const app=document.getElementById('app'), m=D.meta, h=D.headline;
const tone = h.net_return>0?'good':h.net_return<0?'critical':'neutral';

app.appendChild(el(`<div>
  <div class="masthead">
    <div>
      <div class="eyebrow">Backtest debrief · hftbacktest</div>
      <h1>${esc(m.symbol)} — ${esc(m.split)} split</h1>
      <p class="sub">Model signals replayed against the recorded Bybit feed. Equity, the
      pre-fee equity and buy-and-hold are all indexed to percent of book size, so the
      three read on one axis.</p>
    </div>
    <div class="verdict">
      <div class="figure ${tone}">${fmtPct(h.net_return)}</div>
      <div class="cap">Net return on ${m.initial_equity.toLocaleString()} USDT</div>
    </div>
  </div>
</div>`));

app.appendChild(el(`<div class="runbar">
  <span><b>Range</b> ${esc(m.start)} → ${esc(m.end)}</span>
  <span><b>Horizon</b> #${m.horizon_index}</span>
  <span><b>Confidence</b> ≥ ${m.min_confidence}</span>
  <span><b>Queue</b> ${esc(m.queue_model)}</span>
  <span><b>Latency</b> ${esc(m.latency_model)}</span>
  <span><b>Checkpoint</b> ${esc(m.checkpoint)}</span>
</div>`));

app.appendChild(el(`<div class="tiles">${D.metrics.map(x=>
  `<div class="tile"><div class="k">${esc(x.label)}</div>
   <div class="v ${x.tone||''}">${esc(x.value)}</div></div>`).join('')}</div>`));

/* hero chart */
const heroCard=el(`<section class="card">
  <header><h2>Cumulative return</h2>
    <button class="toggle" id="tbl-toggle" aria-expanded="false">Table</button></header>
  <p class="hint">Fees cost ${h.fee_drag.toFixed(2)} percentage points over the run — the gap
  between the two equity lines. Buy-and-hold over the same window returned ${fmtPct(h.benchmark)}.</p>
  <div class="legend">${SERIES.map((s,i)=>
    `<span><i class="swatch" style="background:${S[i]}"></i>${esc(s.name)}</span>`).join('')}</div>
  <div id="hero"></div>
  <div id="hero-table" hidden></div>
</section>`);
app.appendChild(heroCard);

const heroSeries=SERIES.map((s,i)=>({name:s.name,color:S[i],data:D.series[s.key]}));
lineChart(document.getElementById('hero'), heroSeries, {id:'hero',height:300,aria:'Cumulative return over time'});

document.getElementById('tbl-toggle').addEventListener('click',e=>{
  const box=document.getElementById('hero-table'), open=box.hidden;
  e.target.setAttribute('aria-expanded',String(open));
  e.target.textContent = open ? 'Chart' : 'Table';
  document.getElementById('hero').hidden = open;
  box.hidden = !open;
  if(open && !box.dataset.built){
    const step=Math.max(1,Math.floor(D.series.t.length/40));
    const rows=[];
    for(let i=0;i<D.series.t.length;i+=step)
      rows.push(`<tr><td>${fmtTime(D.series.t[i])}</td>${heroSeries.map(s=>
        `<td class="n">${fmtPct(s.data[i])}</td>`).join('')}</tr>`);
    box.innerHTML=`<div style="overflow-x:auto"><table><thead><tr><th>Time (UTC)</th>${
      heroSeries.map(s=>`<th style="text-align:right">${esc(s.name)}</th>`).join('')
    }</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
    box.dataset.built='1';
  }
});

/* drawdown + position */
const two=el(`<div class="two">
  <section class="card"><header><h2>Drawdown</h2></header>
    <p class="hint">Depth below the running peak. Bucket maxima are kept when the series is
    thinned, so no spike is lost.</p><div id="dd"></div></section>
  <section class="card"><header><h2>Position</h2></header>
    <p class="hint">Contracts held. One position at a time, by construction.</p>
    <div id="pos"></div></section>
</div>`);
app.appendChild(two);
lineChart(document.getElementById('dd'),
  [{name:'Drawdown',color:cssv('--critical'),data:D.series.drawdown_pct.map(v=>-v)}],
  {id:'dd',height:190,right:0,aria:'Drawdown over time'});
lineChart(document.getElementById('pos'),
  [{name:'Position',color:cssv('--s1'),data:D.series.position}],
  {id:'pos',height:190,right:0,unit:'',digits:3,aria:'Position over time'});

/* funnel + exits */
const ORDINAL=['#86b6ef','#6da7ec','#3987e5','#2a78d6','#256abf','#1c5cab'];
const fx=el(`<div class="two">
  <section class="card"><header><h2>Decision funnel</h2></header>
    <p class="hint">Where intent is lost between the model and the book. hftbacktest cannot
    see these — they are the strategy's own counters, and they say whether a weak result is
    the model's fault or the execution's.</p><div id="funnel"></div></section>
  <section class="card"><header><h2>Why positions closed</h2></header>
    <p class="hint">Hold expiry means the trained horizon ran out; the protective legs
    firing often means the bracket is tighter than the signal's edge.</p>
    <div id="exits"></div></section>
</div>`);
app.appendChild(fx);
bars(document.getElementById('funnel'), D.funnel, null, ORDINAL);
bars(document.getElementById('exits'),
     D.exits.length?D.exits:[{label:'No positions closed',value:0}], cssv('--s2'));

/* configuration */
app.appendChild(el(`<section class="card"><header><h2>Run configuration</h2></header>
  <div style="overflow-x:auto"><table><tbody>
  <tr><th>Order notional</th><td class="n">${m.order_notional} USDT</td>
      <th>Book size</th><td class="n">${m.initial_equity.toLocaleString()} USDT</td></tr>
  <tr><th>Hold</th><td class="n">${m.hold_ms} ms</td>
      <th>Take profit / stop loss</th><td class="n">${m.take_profit_bps} / ${m.stop_loss_bps} bps</td></tr>
  <tr><th>Maker fee</th><td class="n">${(m.maker_fee_rate*10000).toFixed(2)} bps</td>
      <th>Taker fee</th><td class="n">${(m.taker_fee_rate*10000).toFixed(2)} bps</td></tr>
  <tr><th>Queue model</th><td class="n">${esc(m.queue_model)}</td>
      <th>Latency model</th><td class="n">${esc(m.latency_model)}</td></tr>
  </tbody></table></div></section>`));

D.warnings.forEach(w=>app.appendChild(el(`<div class="note"><b>Assumption.</b> ${esc(w)}</div>`)));

addEventListener('resize',()=>{
  lineChart(document.getElementById('hero'),heroSeries,{id:'hero',height:300,aria:'Cumulative return over time'});
  lineChart(document.getElementById('dd'),[{name:'Drawdown',color:cssv('--critical'),data:D.series.drawdown_pct.map(v=>-v)}],{id:'dd',height:190,right:0,aria:'Drawdown over time'});
  lineChart(document.getElementById('pos'),[{name:'Position',color:cssv('--s1'),data:D.series.position}],{id:'pos',height:190,right:0,unit:'',digits:3,aria:'Position over time'});
});
</script>
"""


if __name__ == "__main__":
    main()
