# ml01-top8-alphas

Top 8 of the 39 single grammar alphas paper-traded in ML01 (`SINGLES_OS10`), ranked by paper return
2026-10-13 .. 2026-10-18 (6 days). Only 3 of the 39 were positive over that window, so the ranking
is mostly noise; treat this as a snapshot, not a selection.

## Alphas

| # | Key | IS Sharpe (2021-23) | OS Sharpe (2024-01..2026-06) | Paper return |
|---|-----|---------------------|------------------------------|--------------|
| 1 | `diverge\|fund_chg40\|taker20\|s5` | 1.85 | 1.34 | +1.05% |
| 2 | `cond\|taker10\|taker20\|low\|s3` | 1.47 | 1.04 | +0.66% |
| 3 | `regime\|illiq20\|taker40\|dispersion\|s5` | 2.15 | 1.68 | +0.11% |
| 4 | `regime\|vpin60\|smart_dumb10\|mkt_vol\|s5` | 1.55 | 1.08 | -0.11% |
| 5 | `diverge\|trade_size10\|vwap_dev40\|s0` | 1.56 | 1.48 | -0.62% |
| 6 | `diverge\|size60\|taker5\|s0` | 2.54 | 1.00 | -1.10% |
| 7 | `cond\|trade_size60\|trade_size5\|low\|s0` | 1.66 | 1.77 | -1.41% |
| 8 | `diverge\|size5\|taker5\|s5` | 2.97 | 1.58 | -1.45% |

Expressions (sign +1: long high score, short low score):

1. `diverge|fund_chg40|taker20|s5`
   ```
   cs_zscore(ts_ema(cs_zscore(ts_mean(funding, 3) - ts_mean(funding, 40)) - cs_zscore(ts_mean(taker_buy_ratio, 20)), 5))
   ```
2. `cond|taker10|taker20|low|s3`
   ```
   cs_zscore(ts_ema(cs_zscore(ts_mean(taker_buy_ratio, 10)) * (1 - rank(ts_mean(taker_buy_ratio, 20))), 3))
   ```
3. `regime|illiq20|taker40|dispersion|s5`
   ```
   cs_zscore(ts_ema(where(st_dispersion > 0, cs_zscore(log(ts_mean(abs(returns) / quote_volume, 20))), cs_zscore(ts_mean(taker_buy_ratio, 40))), 5))
   ```
4. `regime|vpin60|smart_dumb10|mkt_vol|s5`
   ```
   cs_zscore(ts_ema(where(st_mkt_vol > 0, cs_zscore(ts_mean(ag_vpin, 60)), cs_zscore(log(ts_mean(ls_top_pos, 10)) - log(ts_mean(ls_global, 10)))), 5))
   ```
5. `diverge|trade_size10|vwap_dev40|s0`
   ```
   cs_zscore(cs_zscore(log(ts_mean(trade_size, 10))) - cs_zscore(ts_mean(log(close / vwap), 40)))
   ```
6. `diverge|size60|taker5|s0`
   ```
   cs_zscore(cs_zscore(log(ts_mean(quote_volume, 60))) - cs_zscore(ts_mean(taker_buy_ratio, 5)))
   ```
7. `cond|trade_size60|trade_size5|low|s0`
   ```
   cs_zscore(cs_zscore(log(ts_mean(trade_size, 60))) * (1 - rank(log(ts_mean(trade_size, 5)))))
   ```
8. `diverge|size5|taker5|s5`
   ```
   cs_zscore(ts_ema(cs_zscore(log(ts_mean(quote_volume, 5))) - cs_zscore(ts_mean(taker_buy_ratio, 5)), 5))
   ```

## Setup

| Item | Value |
|------|-------|
| Universe | Binance USDT-M perps, point-in-time incl. delisted, top 200 by 30d $ADV re-picked every 30 d |
| Signal / fill | signal from data to close d-1, fill at 01:00 UTC d, mark at 01:00 UTC d+1 |
| Costs | 5 bp taker + 2/5/10/20 bp tier by prior-day $ volume, real funding |
| Account | $10k per alpha, frozen 2026-10-12, first fill 2026-10-13 |
| Selection | OS Sharpe >= 1.0 and IS Sharpe >= 1.4 (OS already seen, so OS is not a holdout for this set) |

## Files

| File | Content |
|------|---------|
| `alphas.json` | key, expression, sign, IS / OS Sharpe, paper return and rank |
| `paper_daily.csv` | daily paper rows of the 8 alphas (return, equity, pnl, funding, cost, n, exposures) |
| `grammar.py` | the typed grammar: primitives, templates, `parse_key` / `expr` (stdlib only) |

Data panels and the DSL engine are not included.
