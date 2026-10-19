"""
Live trader for the 8 alphas of this repo: one Binance USDT-M account (sub-account) per alpha, one-way mode, cross margin.
Signal : the ML01 paper engine (scripts/paper_frozen._grammar_env + _alpha_weights): PIT panels truncated at close d-1,
         PIT top-200 basket re-picked every 30 days. book "paper" = the paper / blackbox weights (cs_scale of
         winsor(zscore, 3), every basket name, gross 1); book "topK" = K highest long +0.5/K, K lowest short -0.5/K.
Data   : run at 00:0x UTC = 07:00 local, bar d-1 comes from REST daily klines (OHLCV, quote volume, trade count, taker
         buy quote volume); funding from panel_pit/funding.parquet. After the 08:15 local data refresh the panel has
         bar d-1 and is used as is. ag_* / ls_* (aggTrades / metrics) are published a day late, so they are always
         lagged 1 day: blackbox variant B, Sharpe within 0.07 of variant A for the one alpha that reads them.
Sizing : fixed gross notional per alpha (config.json) -> |w| * notional per name; names below the exchange minimum
         are skipped (the log prints the skipped share of gross).
Run    : idempotent per (alpha, data bar); orders are diffs vs the account, so a retry run only places what is missing.

  python live/run_live.py                 # DRY RUN (default): prints every alpha's orders, sends nothing
  TOP8_LIVE=1 python live/run_live.py     # sends market orders for enabled alphas that have keys
  python live/run_live.py --only a1,a4    # subset of slots
  python live/run_live.py check           # REST-built vs panel-built weights for the last panel bar (parity test)
  python live/run_live.py ledger          # only sync every account's records to ledger/<slot>/
Env (live/.env): BINANCE_KEY_<SLOT> / BINANCE_SECRET_<SLOT> per slot (e.g. BINANCE_KEY_A1), TOP8_LIVE, ML01_ROOT.
Python: the ML01 venv (~/ML01/.venv/bin/python): pandas, ccxt, python-dotenv and ML01 itself.
"""
import json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
from dotenv import dotenv_values, load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
ML01 = Path(os.getenv("ML01_ROOT", Path.home() / "ML01")).expanduser()
RES = ML01 / "research/alpha_f_backtest"
PIT = Path.home() / "Desktop/datacryp/derived/panel_pit/1d"
CFG = json.loads((HERE / "config.json").read_text())
LIVE = os.getenv("TOP8_LIVE", "0") == "1"
STATE, LEDGER = HERE / "state", HERE / "ledger"
TELE_DIR = Path(os.getenv("TELE_DIR", Path.home() / "Desktop/alpha-live/system")).expanduser()
OLD_ENV = Path.home() / "Desktop/alpha-live/system/.env"    # account of run_live_single.py: never trade it from here

# same field list as paper_frozen._grammar_env
EXTRA = ("taker_buy_quote_volume", "count", "oi_value", "ls_top_acct", "ls_top_pos", "ls_global", "taker_ls", "spot_close",
         "prem_close", "prem_high", "prem_low", "i_rv", "i_rskew", "i_rkurt", "i_downvar", "i_jump", "i_ac1", "i_roll",
         "r_asia", "r_eu", "r_us", "vs_asia", "vs_eu", "vs_us", "tk_asia", "tk_eu", "tk_us", "tk_late",
         "ag_lg", "ag_lgimb", "ag_smimb", "ag_vpin", "ag_lam", "ag_sweep", "ag_burst")
LAG = ("ag_", "ls_")                                         # published one day late -> lagged 1 day (variant B)
KLINE = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5, "quote_volume": 7, "count": 8, "taker_buy_quote_volume": 10}

(HERE / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [top8] %(message)s",
                    handlers=[logging.FileHandler(HERE / "logs/live.log"), logging.StreamHandler()])
log = logging.getLogger("top8")


def retry(fn, *a, tries=6, wait=30, **k):
    """Binance REST call with back-off: the ML01 data refresh and the other live trader share this IP."""
    for i in range(tries):
        try:
            return fn(*a, **k)
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
            if i == tries - 1:
                raise
            log.warning(f"  {type(e).__name__}: retry in {wait}s ({str(e)[:80]})")
            time.sleep(wait)


# --------------------------------------------------------------------------- signal
def _ml01():
    os.chdir(RES)
    for p in (str(ML01 / "scripts"), str(ML01), str(RES)):    # RES first: its engine.py, not ML01/engine/
        if p not in sys.path:
            sys.path.insert(0, p)
    import live6m as L                                         # loads PIT panels at import
    return L


def rest_bar(L, s):
    """Daily kline of bar s (closed) for every panel coin still listed: DataFrame symbol x KLINE fields."""
    ex = ccxt.binance({"options": {"defaultType": "future"}, "enableRateLimit": True})
    listed = {x["symbol"] for x in retry(ex.fapiPublicGetExchangeInfo)["symbols"]}
    cols = L.close.columns
    live = [c for c in cols if c in listed and L.close[c].loc[:s - pd.Timedelta(days=1)].iloc[-5:].notna().any()]
    t0, rows = int(s.timestamp() * 1000), {}
    for c in live:
        k = retry(ex.fapiPublicGetKlines, {"symbol": c, "interval": "1d", "startTime": t0, "limit": 1})
        if k and int(k[0][0]) == t0 and int(k[0][6]) < ex.milliseconds():
            rows[c] = [float(k[0][i]) for i in KLINE.values()]
    new = pd.DataFrame.from_dict(rows, orient="index", columns=list(KLINE))
    log.info(f"  bar {s.date()} from REST klines: {len(new)}/{len(live)} names")
    return new[new.volume > 0]                                  # the panel drops zero-volume bars


def market_env(today, source="auto"):
    """(env, mask, bar, source) for trading day `today` from data up to close today-1 (bar s)."""
    L = _ml01()
    import repo_pit as R
    from alpha_lab.dsl import build_env
    s = today - pd.Timedelta(days=1)
    last = L.close.index.max()
    if source == "auto":
        source = "panel" if last >= s else "rest" if last == s - pd.Timedelta(days=1) else None
    if source is None:
        raise RuntimeError(f"daily panel last bar {last.date()} is older than {(s - pd.Timedelta(days=1)).date()}: "
                           "data refresh not done")
    new = rest_bar(L, s) if source == "rest" else None
    cols = L.close.columns
    idx = L.close.index[L.close.index <= s].union(pd.DatetimeIndex([s]))

    def field(df, f):
        x = df.astype(float).reindex(columns=cols)
        if new is not None:                                     # REST row replaces / appends bar s
            x = x.loc[:s - pd.Timedelta(days=1)]
            if f in new:
                x = pd.concat([x, new[f].reindex(cols).to_frame(s).T])
        x = x.reindex(idx)
        return x.shift(1) if f.startswith(LAG) else x

    P = {k: field(pd.read_parquet(R.PD / f"d_{k}.parquet"), k) for k in ("open", "high", "low")}
    P.update(close=field(L.close, "close"), volume=field(L.vol, "volume"), quote_volume=field(L.qv, "quote_volume"))
    for f in EXTRA:
        if (PIT / f"{f}.parquet").exists():
            P[f] = field(pd.read_parquet(PIT / f"{f}.parquet"), f)
    env = build_env(P)
    mem, _ = L.basket(P["close"], P["quote_volume"], today, 30)
    mask = pd.Series(False, index=cols)
    mask[list(mem & set(cols))] = True
    return env, mask & P["close"].iloc[-1].notna(), str(s.date()), source


def weights(a, env, mask):
    """{symbol: weight}, gross 1, for one alpha; None when too few names carry a signal (data not there yet)."""
    from alpha_lab import grammar as G
    from alpha_lab.dsl import evaluate
    from alpha_lab.newalpha import _weights
    from alpha_lab.operators import cs_winsorize, cs_zscore
    raw = evaluate(G.expr(G.parse_key(a["key"])), env).iloc[-1:].replace([np.inf, -np.inf], np.nan)
    raw = raw.where(mask.to_frame().T.values)
    n = int(raw.iloc[-1].notna().sum())
    if n < 0.5 * mask.sum():
        log.warning(f"  {a['slot']}: signal on {n}/{int(mask.sum())} basket names -> skipped this run")
        return None
    book = a.get("book", "paper")
    if book == "paper":                                         # = paper_frozen._alpha_weights
        w = (a["sign"] * _weights(raw, "norank", 0.1)).iloc[-1].fillna(0)
    else:
        k = int(book.removeprefix("top"))
        z = (a["sign"] * cs_winsorize(cs_zscore(raw), 3)).iloc[-1].dropna()
        w = pd.concat([pd.Series(0.5 / k, z.nlargest(k).index), pd.Series(-0.5 / k, z.nsmallest(k).index)])
    w = w[w.abs() > 1e-9]
    return (w / w.abs().sum()).to_dict()


# --------------------------------------------------------------------------- exchange
def keys(slot):
    return os.getenv(f"BINANCE_KEY_{slot.upper()}", ""), os.getenv(f"BINANCE_SECRET_{slot.upper()}", "")


def check_keys(slots):
    """Each slot needs its own account: refuse duplicated keys and the key of the existing single-alpha trader."""
    seen = {}
    old = dotenv_values(OLD_ENV).get("BINANCE_API_KEY") if OLD_ENV.exists() else None
    for s in slots:
        k = keys(s)[0]
        if not k:
            continue
        if k == old:
            raise RuntimeError(f"{s}: key is the run_live_single.py account; every alpha needs its own sub-account")
        if k in seen:
            raise RuntimeError(f"{s} and {seen[k]} share one API key; positions would net across alphas")
        seen[k] = s


def exchange(slot=None):
    k, sec = keys(slot) if slot else ("", "")
    ex = ccxt.binance({"apiKey": k, "secret": sec, "enableRateLimit": True,
                       "options": {"defaultType": "future", "fetchCurrencies": False, "fetchMarkets": {"types": ["linear"]}}})
    retry(ex.load_markets)
    return ex


def account(ex):
    """Wallet, available balance and positions {ccxt symbol: signed qty}; checks one-way position mode."""
    if ex.fapiPrivateGetPositionSideDual()["dualSidePosition"]:
        raise RuntimeError("account is in hedge mode; switch to one-way mode")
    acc = ex.fapiPrivateV2GetAccount()
    pos = {}
    for p in acc["positions"]:
        amt = float(p["positionAmt"])
        if amt != 0:
            pos[ex.safe_symbol(p["symbol"], None, None, "swap")] = amt
            if p["isolated"]:
                log.warning(f"  {p['symbol']} is ISOLATED margin")
    return float(acc["totalWalletBalance"]), float(acc["availableBalance"]), pos


def _qty(ex, s, x):
    """Amount truncated to the symbol's step; 0 when below one step."""
    try:
        return float(ex.amount_to_precision(s, x))
    except ccxt.InvalidOrder:
        return 0.0


def _min_cost(ex, s):
    return (ex.market(s)["limits"]["cost"] or {}).get("min") or 5.0


def plan(ex, w, pos, prices, notional):
    """Market orders moving pos -> target. Reductions / closes first (they free margin), then opens / adds."""
    tgt, skipped = {}, 0.0
    for s, wt in w.items():
        m = ex.safe_symbol(s, None, None, "swap")
        if m not in ex.markets or m not in prices:
            log.warning(f"    skip {s}: not tradable on the exchange")
            skipped += abs(wt); continue
        px = prices[m]
        q = _qty(ex, m, notional * abs(wt) / px)
        if q * px < _min_cost(ex, m) or q < ((ex.market(m)["limits"]["amount"] or {}).get("min") or 0):
            skipped += abs(wt); continue
        tgt[m] = q if wt > 0 else -q
    orders = []
    for s in set(tgt) | set(pos):
        cur, new = pos.get(s, 0.0), tgt.get(s, 0.0)
        d = new - cur
        if d == 0:
            continue
        reduce = abs(new) < abs(cur) and np.sign(new) in (0, np.sign(cur))
        if reduce or new == 0:
            q = _qty(ex, s, abs(d))
            if q > 0:
                orders.append(dict(symbol=s, side="buy" if d > 0 else "sell", qty=q, reduce=True, usd=q * prices.get(s, 0)))
        else:
            if cur != 0 and np.sign(new) != np.sign(cur):                 # flip: close first, then open
                orders.append(dict(symbol=s, side="buy" if cur < 0 else "sell", qty=abs(cur), reduce=True, usd=abs(cur) * prices.get(s, 0)))
                d = new
            q = _qty(ex, s, abs(d))
            if q * prices[s] >= _min_cost(ex, s):
                orders.append(dict(symbol=s, side="buy" if d > 0 else "sell", qty=q, reduce=False, usd=q * prices[s]))
    return sorted(orders, key=lambda o: not o["reduce"]), tgt, skipped


def send(ex, o, leverage):
    if not o["reduce"]:
        try: ex.set_margin_mode("cross", o["symbol"])
        except Exception as e:
            if "No need to change" not in str(e): log.warning(f"    margin mode {o['symbol']}: {str(e)[:80]}")
        try: ex.set_leverage(leverage, o["symbol"])
        except Exception as e:
            log.warning(f"    leverage {o['symbol']}: {str(e)[:80]}")
            if "-4028" in str(e):                                  # symbol max leverage < leverage: use its max
                b = ex.fapiPrivateGetLeverageBracket({"symbol": ex.market(o["symbol"])["id"]})[0]["brackets"]
                ex.set_leverage(int(b[0]["initialLeverage"]), o["symbol"])
    return ex.create_order(o["symbol"], "market", o["side"], o["qty"],
                           params={**({"reduceOnly": True} if o["reduce"] else {}), "newOrderRespType": "RESULT"})


# --------------------------------------------------------------------------- ledger
def _append(slot, name, rows, key):
    """Append rows to ledger/<slot>/<name>.csv, skipping ids already saved (every sync is idempotent)."""
    if not rows:
        return 0
    d = LEDGER / slot
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.csv"
    seen = set(pd.read_csv(f, usecols=[key], dtype=str)[key]) if f.exists() else set()
    new = [r for r in rows if str(r[key]) not in seen]
    if new:
        pd.DataFrame(new).to_csv(f, mode="a", header=not f.exists(), index=False)
    return len(new)


def sync_ledger(ex, slot, bar, start):
    """income.csv (funding, commission, realized PnL per trade; account-wide, no per-symbol calls) + account.csv and
    positions.csv snapshots. Per-symbol fills are not pulled: ~190 names x 8 accounts would exceed the IP weight."""
    f = LEDGER / slot / "income.csv"
    since = max(start, int(pd.read_csv(f, usecols=["time"])["time"].max()) - 3_600_000) if f.exists() else start
    now, inc = ex.milliseconds(), []
    while since < now:                                           # the income endpoint serves at most 7 days per call
        end = min(since + 7 * 86_400_000 - 1, now)
        r = retry(ex.fapiPrivateGetIncome, {"startTime": since, "endTime": end, "limit": 1000})
        inc += [dict(income_id=f"{x['tranId']}-{x['incomeType']}-{x['symbol']}", time=int(x["time"]),
                     utc=ex.iso8601(int(x["time"])), symbol=x["symbol"], type=x["incomeType"], amount=float(x["income"]),
                     asset=x["asset"], trade_id=x.get("tradeId", "")) for x in r]
        since = int(r[-1]["time"]) + 1 if len(r) == 1000 else end + 1
    acc = retry(ex.fapiPrivateV2GetAccount)
    t = ex.milliseconds()
    P = [p for p in acc["positions"] if float(p["positionAmt"]) != 0]
    snap = dict(snap_id=str(t), time=t, utc=ex.iso8601(t), data_bar=bar, wallet=float(acc["totalWalletBalance"]),
                margin_balance=float(acc["totalMarginBalance"]), available=float(acc["availableBalance"]),
                init_margin=float(acc["totalInitialMargin"]), upnl=float(acc["totalUnrealizedProfit"]), n_pos=len(P),
                long_usd=sum(float(p["notional"]) for p in P if float(p["notional"]) > 0),
                short_usd=sum(float(p["notional"]) for p in P if float(p["notional"]) < 0))
    prow = [dict(row_id=f"{t}-{p['symbol']}", time=t, utc=ex.iso8601(t), symbol=p["symbol"], qty=float(p["positionAmt"]),
                 entry=float(p["entryPrice"]), notional=float(p["notional"]), upnl=float(p["unrealizedProfit"])) for p in P]
    n = (_append(slot, "income", inc, "income_id"), _append(slot, "account", [snap], "snap_id"),
         _append(slot, "positions", prow, "row_id"))
    log.info(f"  {slot} ledger: +{n[0]} income, account ${snap['margin_balance']:.2f}, {len(P)} positions")


def alert(text):
    """Telegram alert through alpha-live's tele_report (same bot); never blocks trading."""
    try:
        sys.path.insert(0, str(TELE_DIR))
        import tele_report
        tele_report.alert(f"[top8] {text}")
    except Exception as e:
        log.error(f"  telegram alert failed: {str(e)[:120]}")


# --------------------------------------------------------------------------- main
def trade(a, w, bar, pub, prices):
    """Bring one alpha's account to its target book. Returns 'done' / 'partial' / 'skip' / 'dry'."""
    slot, notional, lev = a["slot"], float(a["notional"]), int(a["leverage"])
    sf = STATE / f"{slot}.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    has_key = bool(keys(slot)[0])
    live = LIVE and a.get("enabled") and has_key
    if live and st.get("data_bar") == bar:
        log.info(f"  {slot}: bar {bar} already traded at {st['at']}")
        return "skip"
    ex = exchange(slot) if has_key else pub
    if has_key:
        wallet, avail, pos = retry(account, ex)
    else:
        wallet, avail, pos = float("nan"), float("nan"), {}
    orders, tgt, skipped = plan(ex, w, pos, prices, notional)
    gross = sum(abs(q) * prices[s] for s, q in tgt.items())
    log.info(f"  {slot} {a['key']} [{a.get('book', 'paper')}] {'LIVE' if live else 'DRY'} | wallet ${wallet:.2f} | "
             f"{len(tgt)} names gross ${gross:.0f} of ${notional:.0f} (skipped {skipped:.1%} below exchange minimum) | "
             f"{len(orders)} orders, {sum(o['usd'] for o in orders):.0f} USD")
    if not live:
        for o in orders[:5]:
            log.info(f"    [DRY] {o['side']:4s} {o['qty']} {o['symbol']} (${o['usd']:.1f}){' reduceOnly' if o['reduce'] else ''}")
        if len(orders) > 5:
            log.info(f"    [DRY] ... {len(orders) - 5} more")
        return "dry"
    need = notional / lev
    if wallet < 2 * need:
        raise RuntimeError(f"{slot}: wallet ${wallet:.0f} < 2 x margin ${need:.0f}, refusing to trade")
    ok, sent = 0, []
    for o in orders:
        try:
            r = send(ex, o, lev)
            ok += 1; time.sleep(0.15)
            sent.append(dict(order_id=r["id"], time=r["timestamp"] or ex.milliseconds(), data_bar=bar, symbol=r["info"]["symbol"],
                             side=o["side"], qty=o["qty"], reduce_only=o["reduce"], status=r["status"], filled=r["filled"],
                             avg_price=r["average"], usd=(r["filled"] or 0) * (r["average"] or 0), error=""))
        except Exception as e:
            log.error(f"    {slot} FAIL {o['symbol']}: {str(e)[:160]}")
            t = ex.milliseconds()
            sent.append(dict(order_id=f"fail-{t}-{o['symbol']}", time=t, data_bar=bar, symbol=o["symbol"], side=o["side"],
                             qty=o["qty"], reduce_only=o["reduce"], status="error", filled=0, avg_price=None, usd=None,
                             error=str(e)[:300]))
    log.info(f"  {slot}: executed {ok}/{len(orders)}")
    try:
        for x in sent:
            x["utc"] = ex.iso8601(x["time"])
        _append(slot, "orders", sent, "order_id")
        sync_ledger(ex, slot, bar, st.get("first_ms") or ex.milliseconds() - 86_400_000)
    except Exception as e:                                       # the book must never block trading
        log.error(f"  {slot} ledger sync failed: {str(e)[:200]}")
    if ok < len(orders):
        bad = ", ".join(x["symbol"] for x in sent if x["status"] == "error")
        alert(f"{slot} bar {bar}: khớp {ok}/{len(orders)} lệnh. Lỗi: {bad}. Lần chạy kế tiếp sẽ đặt lại phần còn thiếu.")
        return "partial"
    STATE.mkdir(exist_ok=True)
    sf.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), data_bar=bar, key=a["key"], notional=notional,
                                  wallet=wallet, executed=ok, orders=len(orders),
                                  first_ms=st.get("first_ms") or int(sent[0]["time"]) if sent else st.get("first_ms")), indent=1))
    return "done"


def check():
    """Parity test: weights for the day after the last panel bar, built from the panel vs from REST klines."""
    L = _ml01()
    today = L.close.index.max() + pd.Timedelta(days=1)
    e1, m1, bar, _ = market_env(today, "panel")
    e2, m2, _, _ = market_env(today, "rest")
    log.info(f"check bar {bar}: basket panel {int(m1.sum())} / rest {int(m2.sum())}")
    for a in CFG["alphas"]:
        w1, w2 = weights(a, e1, m1), weights(a, e2, m2)
        if w1 is None or w2 is None:
            log.info(f"  {a['slot']}: no signal (panel {w1 is not None}, rest {w2 is not None})"); continue
        d = pd.Series(w1).sub(pd.Series(w2), fill_value=0).abs()
        log.info(f"  {a['slot']} {a['key']}: sum|dw| {d.sum():.4f}, max {d.max():.4f}, names {len(w1)} / {len(w2)}")


def main():
    args = sys.argv[1:]
    only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
    alphas = [a for a in CFG["alphas"] if only is None or a["slot"] in only]
    check_keys([a["slot"] for a in CFG["alphas"]])
    if "check" in args:
        return check()
    if "ledger" in args:
        for a in alphas:
            if keys(a["slot"])[0]:
                st = json.loads((STATE / f"{a['slot']}.json").read_text()) if (STATE / f"{a['slot']}.json").exists() else {}
                ex = exchange(a["slot"])
                sync_ledger(ex, a["slot"], st.get("data_bar"), st.get("first_ms") or ex.milliseconds() - 86_400_000)
        return
    log.info("=" * 70)
    log.info(f"{len(alphas)} alphas | {'LIVE' if LIVE else 'DRY RUN'}")
    today = pd.Timestamp.now(tz="UTC").normalize()
    env, mask, bar, src = market_env(today)
    log.info(f"  signal from bar {bar} ({src}), basket {int(mask.sum())} names")
    pub = exchange()
    W = {a["slot"]: weights(a, env, mask) for a in alphas}
    syms = sorted({pub.safe_symbol(s, None, None, "swap") for w in W.values() if w for s in w} & set(pub.markets))
    prices = {s: float(t["last"]) for s, t in retry(pub.fetch_tickers, syms).items()} if syms else {}
    res = {}
    for a in alphas:
        if W[a["slot"]] is None:
            res[a["slot"]] = "no-signal"; continue
        try:
            res[a["slot"]] = trade(a, W[a["slot"]], bar, pub, prices)
        except Exception as e:                                   # one account failing must not stop the others
            log.error(f"  {a['slot']} ERROR: {str(e)[:300]}")
            res[a["slot"]] = "error"
            if LIVE and a.get("enabled"):
                alert(f"{a['slot']} lỗi: <code>{str(e)[:300]}</code>")
    log.info(f"  result: {res}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if LIVE:
            alert(f"Lỗi: <code>{str(e)[:300]}</code>")
        raise
