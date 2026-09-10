from dataclasses import dataclass, field
from decimal import Decimal

# What the loop was able to do on one pass. The first four mean no prediction was
# made at all, which is itself worth measuring: a bot that is never in a position
# to act has a data or warm-up problem, not a model problem.
NO_BOOK = "NO_BOOK"
NOT_READY = "NOT_READY"
HOLDING = "HOLDING"
WAITING = "WAITING"
BLOCKED = "BLOCKED"
REJECTED = "REJECTED"
APPROVED = "APPROVED"

PREDICTED_STATUSES = frozenset({BLOCKED, REJECTED, APPROVED})


@dataclass
class PipelineCounters:
    """Funnel counters, shared by the live loop and the backtest.

    Deliberately one type for both: run a backtest over the window a live session
    covered, subtract one funnel from the other, and the stage where they diverge
    is the modelling error. Separate types would make that comparison guesswork.

    `signal_blocked` and `risk_rejected` key on the reasons the signal policy and
    the risk gate already return, so a rejection is never just a missing count.
    """

    decisions: int = 0
    not_ready: int = 0
    no_book: int = 0
    signals_actionable: int = 0
    signals_approved: int = 0
    signal_blocked: dict[str, int] = field(default_factory=dict)
    risk_rejected: dict[str, int] = field(default_factory=dict)
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_expired: int = 0
    exit_repegs: int = 0
    """Times a resting passive exit was re-posted because the book moved away.

    Read against `orders_filled`: re-pegs per trade is how hard the exit had to
    chase, and it is what trades taker fees for holding time.
    """
    exits_by_reason: dict[str, int] = field(default_factory=dict)

    def blocked(self, reason: str) -> None:
        self.signal_blocked[reason] = self.signal_blocked.get(reason, 0) + 1

    def rejected(self, reason: str) -> None:
        self.risk_rejected[reason] = self.risk_rejected.get(reason, 0) + 1

    def exited(self, reason: str) -> None:
        self.exits_by_reason[reason] = self.exits_by_reason.get(reason, 0) + 1

    @property
    def fill_rate(self) -> float:
        return self.orders_filled / self.orders_submitted if self.orders_submitted else 0.0

    @property
    def approval_rate(self) -> float:
        return (
            self.signals_approved / self.signals_actionable
            if self.signals_actionable
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One pass of the decision loop, persisted so it can be scored later.

    The probabilities and the timestamp are what make "predicted vs realized"
    possible: replay the same smooth mid-price label the training set used
    against the snapshots that followed, and the live score becomes directly
    comparable to validation F1.
    """

    timestamp_ms: int
    status: str
    action: str
    horizon_index: int
    p_down: float
    p_flat: float
    p_up: float
    predicted_class: int
    confidence: float
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    book_age_ms: int = 0
    is_reset: bool = False
    inference_us: int = 0
    signal_reason: str | None = None
    risk_reason: str | None = None
    flow_z_max: float = 0.0
    flow_z_abs_mean: float = 0.0
    ctx_ticker_age_ms: int = -1
    ctx_liq_age_ms: int = -1
    ctx_ls_age_ms: int = -1
    class_changed: bool = False


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A step in one order's life, carrying the latency actually observed.

    `latency_ms` is measured from submission, which is the number the backtest
    can only assume — a session of these is what turns `intp_order_latency` from
    a guess into data.
    """

    timestamp_ms: int
    order_link_id: str
    event: str
    side: str
    order_type: str
    reduce_only: bool
    qty: Decimal
    limit_price: Decimal | None = None
    fill_price: Decimal | None = None
    decision_price: Decimal | None = None
    latency_ms: int | None = None
    exit_reason: str | None = None
