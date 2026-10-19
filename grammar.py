"""Typed alpha grammar: scale-free primitives x templates -> DSL expression (2026-10-09).

Why: free-text LLM generation collapsed onto `ts_std(volume)` motifs, used raw token volume / price differences
(meme and price-level artifacts) and wasted most trials on parse / complexity failures. Here the LLM (or a sweep)
only picks slots; code builds the expression, so every candidate is valid, scale-free and has a canonical key.

Rules encoded by construction:
- every primitive is dimensionless (returns, log-ratios, rates, ratios) -> no raw price / token volume in a cs op
- windows >= 5 bars (daily signals with ~1.4 turnover/day die on 7bp costs)
- combination only after cs normalisation (cs_zscore / rank), so units never mix
"""
from __future__ import annotations

import itertools
import random

WINDOWS = (5, 10, 20, 40, 60)
SMOOTH = (0, 3, 5)

# name -> (family, DSL with {w}, description). All dimensionless.
PRIMITIVES = {
    # price path
    "mom": ("price", "ts_log_return(close, {w})", "log return over w days"),
    "dist_high": ("price", "log(close / ts_max(high, {w}))", "distance below w-day high"),
    "dist_low": ("price", "log(close / ts_min(low, {w}))", "distance above w-day low"),
    "vwap_dev": ("price", "ts_mean(log(close / vwap), {w})", "close vs daily vwap, averaged"),
    "clv": ("price", "ts_mean((close - low) / (high - low + close * 0.0001), {w})", "close location in daily range"),
    # risk / tail
    "vol": ("risk", "ts_std(returns, {w})", "return volatility"),
    "skew": ("risk", "ts_skew(returns, {w})", "return skewness"),
    "maxret": ("risk", "ts_max(returns, {w})", "largest daily return (lottery)"),
    "range": ("risk", "ts_mean(log(high / low), {w})", "average log high/low range"),
    # liquidity (dollar based)
    "illiq": ("liquidity", "log(ts_mean(abs(returns) / quote_volume, {w}))", "Amihud illiquidity"),
    "size": ("liquidity", "log(ts_mean(quote_volume, {w}))", "dollar volume size"),
    "dv_shock": ("liquidity", "log(ts_mean(quote_volume, 3) / ts_mean(quote_volume, {w}))", "recent $volume vs w-day"),
    # funding
    "fund": ("funding", "ts_mean(funding, {w})", "average funding rate"),
    "fund_chg": ("funding", "ts_mean(funding, 3) - ts_mean(funding, {w})", "funding change"),
    "fund_z": ("funding", "ts_zscore(funding, {w})", "funding vs own history"),
    # order flow
    "taker": ("flow", "ts_mean(taker_buy_ratio, {w})", "share of volume from aggressive buyers"),
    "taker_chg": ("flow", "ts_mean(taker_buy_ratio, 3) - ts_mean(taker_buy_ratio, {w})", "change in taker buy share"),
    "trade_size": ("flow", "log(ts_mean(trade_size, {w}))", "average $ per trade (whale vs retail)"),
    "trade_size_chg": ("flow", "log(ts_mean(trade_size, 3) / ts_mean(trade_size, {w}))", "change in $ per trade"),
    "trades_shock": ("flow", "log(ts_mean(n_trades, 3) / ts_mean(n_trades, {w}))", "trade count surge"),
    # positioning (Binance metrics, alts from ~2021-12)
    "oi_chg": ("positioning", "log(oi_value / ts_delay(oi_value, {w}))", "open interest growth"),
    "oi_turn": ("positioning", "log(ts_mean(oi_turnover, {w}))", "$volume / open interest"),
    "ls_top": ("positioning", "log(ts_mean(ls_top_pos, {w}))", "top-trader long/short position ratio"),
    "ls_crowd": ("positioning", "log(ts_mean(ls_global, {w}))", "all-account long/short ratio"),
    "smart_dumb": ("positioning", "log(ts_mean(ls_top_pos, {w})) - log(ts_mean(ls_global, {w}))", "top traders vs crowd"),
    "taker_ls": ("positioning", "log(ts_mean(taker_ls, {w}))", "taker buy/sell volume ratio (5m metrics)"),
    # basis
    "basis": ("basis", "ts_mean(basis, {w})", "perp premium over spot"),
    "basis_chg": ("basis", "ts_mean(basis, 3) - ts_mean(basis, {w})", "change in perp premium"),
    # premium index (perp vs index, all perps incl. 1000x / no-spot coins; added 2026-10-09)
    "premium": ("premium", "ts_mean(premium, {w})", "average perp premium over the index"),
    "premium_chg": ("premium", "ts_mean(premium, 3) - ts_mean(premium, {w})", "change in perp premium"),
    "prem_vol": ("premium", "ts_mean(prem_range, {w})", "intraday range of the premium (dislocation)"),
    # intraday (daily features built from 1h bars, scripts/pit_extra.py intraday; 1d only)
    "rvol": ("intraday", "ts_mean(sqrt(i_rv), {w})", "realized vol from hourly returns"),
    "rskew": ("intraday", "ts_mean(i_rskew, {w})", "realized skew of hourly returns"),
    "rkurt": ("intraday", "ts_mean(i_rkurt, {w})", "realized kurtosis of hourly returns"),
    "downvar": ("intraday", "ts_mean(i_downvar, {w})", "share of variance from down hours"),
    "jump": ("intraday", "ts_mean(i_jump, {w})", "largest hourly move vs daily realized vol"),
    "ac_intra": ("intraday", "ts_mean(i_ac1, {w})", "hourly return autocorrelation (trend vs mean-reversion inside the day)"),
    "roll": ("intraday", "ts_mean(i_roll, {w})", "Roll spread from hourly bounce (illiquidity)"),
    "sess_us": ("intraday", "ts_sum(r_us, {w})", "return made in the US session"),
    "sess_asia": ("intraday", "ts_sum(r_asia, {w})", "return made in the Asia session"),
    "sess_gap": ("intraday", "ts_sum(r_us - r_asia, {w})", "US minus Asia session return"),
    "vshare_us": ("intraday", "ts_mean(vs_us, {w})", "share of daily $volume traded in the US session"),
    # session order flow (needs 1h taker)
    "tk_us": ("session_flow", "ts_mean(tk_us, {w})", "aggressive-buy share in the US session"),
    "tk_asia": ("session_flow", "ts_mean(tk_asia, {w})", "aggressive-buy share in the Asia session"),
    "tk_gap": ("session_flow", "ts_mean(tk_us - tk_asia, {w})", "US minus Asia aggressive-buy share"),
    "tk_late": ("session_flow", "ts_mean(tk_late, {w})", "aggressive-buy share in the last 4 hours of the day"),
    # order flow from aggTrades (scripts/agg_extra.py; 1d only, added 2026-10-10)
    "whale": ("orderflow", "ts_mean(ag_lg, {w})", "share of $volume in the day's largest 5% of trades"),
    "whale_imb": ("orderflow", "ts_mean(ag_lgimb, {w})", "buy-minus-sell share among large trades"),
    "retail_imb": ("orderflow", "ts_mean(ag_smimb, {w})", "buy-minus-sell share among small trades"),
    "whale_gap": ("orderflow", "ts_mean(ag_lgimb - ag_smimb, {w})", "large-trade minus small-trade imbalance"),
    "vpin": ("orderflow", "ts_mean(ag_vpin, {w})", "5-min order-flow imbalance per $volume (toxicity)"),
    "impact": ("orderflow", "ts_mean(ag_lam, {w})", "5-min price impact per share of daily $volume"),
    "sweep": ("orderflow", "ts_mean(log(ag_sweep), {w})", "fills per aggressive order (book sweeping)"),
    "burst": ("orderflow", "ts_mean(ag_burst, {w})", "burstiness of 1-min $volume"),
    # residual / beta (computed in dsl._add_residual_and_state_fields)
    "rmom": ("residual", "ts_sum(resid_mkt, {w})", "momentum of the market-residual return"),
    "rmom_btc": ("residual", "ts_sum(resid_btc, {w})", "momentum of the BTC-residual return"),
    "ivol": ("residual", "ts_std(resid_mkt, {w})", "idiosyncratic volatility"),
    "beta": ("residual", "ts_mean(beta_mkt, {w})", "market beta (60-bar, smoothed)"),
    "beta_btc": ("residual", "ts_mean(beta_btc, {w})", "BTC beta (60-bar, smoothed)"),
}

# market states for the regime template (broadcast panels from dsl._add_residual_and_state_fields)
STATES = ("btc_trend", "mkt_vol", "dispersion")

# template -> (needs_b, description)
TEMPLATES = {
    "single": (False, "cs_zscore(A)"),
    "cond": (True, "cs_zscore(A) * rank(B) [cond=high] or * (1 - rank(B)) [cond=low]: A only where B is high/low"),
    "diverge": (True, "cs_zscore(A) - cs_zscore(B): A not confirmed by B"),
    "interact": (True, "cs_zscore(A) * cs_zscore(B): A and B agree"),
    "regime": (True, "where(state > 0, cs_zscore(A), cs_zscore(B)): A in one market state, B in the other"),
}


class GrammarError(ValueError):
    pass


def prim(name: str, w: int) -> str:
    if name not in PRIMITIVES:
        raise GrammarError(f"unknown primitive {name}")
    if w not in WINDOWS:
        raise GrammarError(f"window {w} not in {WINDOWS}")
    return PRIMITIVES[name][1].format(w=w)


def canonical(c: dict) -> dict:
    """Normalise a candidate; sign is chosen later on IS, so A-B and B-A are the same signal."""
    t = c.get("template")
    if t not in TEMPLATES:
        raise GrammarError(f"unknown template {t}")
    out = dict(template=t, a=c["a"], wa=int(c["wa"]), smooth=int(c.get("smooth", 0)))
    if out["smooth"] not in SMOOTH:
        raise GrammarError(f"smooth {out['smooth']} not in {SMOOTH}")
    prim(out["a"], out["wa"])
    if TEMPLATES[t][0]:
        out.update(b=c["b"], wb=int(c["wb"]))
        prim(out["b"], out["wb"])
        if (out["a"], out["wa"]) == (out["b"], out["wb"]):
            raise GrammarError("A == B")
        if t == "cond":
            out["cond"] = c.get("cond", "high")
            if out["cond"] not in ("high", "low"):
                raise GrammarError("cond must be high|low")
        elif t == "regime":
            out["state"] = c.get("state", "btc_trend")
            if out["state"] not in STATES:
                raise GrammarError(f"state must be one of {STATES}")
        elif (out["b"], out["wb"]) < (out["a"], out["wa"]):            # symmetric up to sign
            out["a"], out["wa"], out["b"], out["wb"] = out["b"], out["wb"], out["a"], out["wa"]
    return out


def key(c: dict) -> str:
    c = canonical(c)
    parts = [c["template"], f"{c['a']}{c['wa']}"]
    if "b" in c:
        parts.append(f"{c['b']}{c['wb']}")
    if "cond" in c:
        parts.append(c["cond"])
    if "state" in c:
        parts.append(c["state"])
    parts.append(f"s{c['smooth']}")
    return "|".join(parts)


def parse_key(k: str) -> dict:
    """Inverse of key(): 'cond|mom20|illiq20|low|s0' -> candidate dict."""
    import re
    parts = k.split("|")
    split = lambda x: (re.match(r"(.+?)(\d+)$", x).group(1), int(re.match(r"(.+?)(\d+)$", x).group(2)))
    c = dict(template=parts[0], smooth=int(parts[-1][1:]))
    c["a"], c["wa"] = split(parts[1])
    if TEMPLATES[c["template"]][0]:
        c["b"], c["wb"] = split(parts[2])
        if c["template"] == "cond":
            c["cond"] = parts[3]
        if c["template"] == "regime":
            c["state"] = parts[3]
    return canonical(c)


def expr(c: dict) -> str:
    c = canonical(c)
    a = f"cs_zscore({prim(c['a'], c['wa'])})"
    t = c["template"]
    if t == "single":
        core = a
    else:
        b_raw = prim(c["b"], c["wb"])
        if t == "cond":
            core = f"{a} * rank({b_raw})" if c["cond"] == "high" else f"{a} * (1 - rank({b_raw}))"
        elif t == "diverge":
            core = f"{a} - cs_zscore({b_raw})"
        elif t == "regime":
            core = f"where(st_{c['state']} > 0, {a}, cs_zscore({b_raw}))"
        else:
            core = f"{a} * cs_zscore({b_raw})"
    return f"cs_zscore(ts_ema({core}, {c['smooth']}))" if c["smooth"] else f"cs_zscore({core})"


def families(c: dict) -> tuple:
    c = canonical(c)
    return tuple(sorted({PRIMITIVES[c["a"]][0]} | ({PRIMITIVES[c["b"]][0]} if "b" in c else set())))


# sampling preferences from the 2026-10-09 OS of 5,046 candidates: single / cond held up best, interact worst,
# unsmoothed signals cost 25-35%/yr in the live-sim -> new candidates are always smoothed
# rule v3 (2026-10-10, OS-informed): over batches 1-6 (276 picks) regime OS median 0.60, diverge 0.55, cond 0.17,
# interact 0.02 -> sampling shifted to regime / diverge; selection rule for batch OS unchanged
TEMPLATE_WEIGHTS = {"single": 0.05, "cond": 0.20, "diverge": 0.35, "interact": 0.05, "regime": 0.35}
SMOOTH_NEW = (3, 5)


def random_candidate(rng: random.Random, available: set[str] | None = None) -> dict:
    names = [n for n in PRIMITIVES if available is None or n in available]
    t = rng.choices(list(TEMPLATE_WEIGHTS), weights=list(TEMPLATE_WEIGHTS.values()))[0]
    c = dict(template=t, a=rng.choice(names), wa=rng.choice(WINDOWS), smooth=rng.choice(SMOOTH_NEW))
    if TEMPLATES[t][0]:
        c.update(b=rng.choice(names), wb=rng.choice(WINDOWS), cond=rng.choice(("high", "low")), state=rng.choice(STATES))
    return c


def singles_sweep(available: set[str] | None = None):
    for n, w, s in itertools.product(PRIMITIVES, WINDOWS, SMOOTH):
        if available is None or n in available:
            yield dict(template="single", a=n, wa=w, smooth=s)


def catalog_text(available: set[str] | None = None) -> str:
    lines = ["PRIMITIVES (name [family]: meaning):"]
    for n, (fam, _, d) in PRIMITIVES.items():
        if available is None or n in available:
            lines.append(f"  {n} [{fam}]: {d}")
    lines.append(f"WINDOWS: {list(WINDOWS)}   SMOOTH (ema of final signal): {list(SMOOTH)}")
    lines.append("TEMPLATES:")
    lines += [f"  {t}: {d}" for t, (_, d) in TEMPLATES.items()]
    lines.append(f"STATES (regime template only): {list(STATES)}")
    return "\n".join(lines)
