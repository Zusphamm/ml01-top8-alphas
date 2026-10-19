# live

`run_live.py` trades each of the 8 alphas in its own Binance USDT-M account (sub-account), daily, with the same
signal code as the ML01 paper engine. Needs `~/ML01` (panels, DSL) and its venv.

## Parity checks (2026-10-19, bar 2026-10-18)

| Check | Result |
|-------|--------|
| Target weights vs `paper_frozen._alpha_weights` (panel path) | identical for 7 / 8; a4 differs by 3.2% of gross (ag / ls lag below) |
| REST-kline bar vs panel bar (`run_live.py check`) | identical weights for all 8 (fields: OHLCV, quote volume, count, taker buy) |
| Basket | same 200 names as the paper engine |

## Data timing

| Run (local) | Source of bar d-1 | Note |
|-------------|-------------------|------|
| 07:02 | REST daily klines (~1 min, 523 calls) | trades right after the 00:00 UTC close |
| 07:32 | same | retry of missing orders |
| 09:22 | panel after the 08:15 data refresh | fallback |

`ag_*` / `ls_*` (aggTrades / metrics) are published a day late, so live always lags them 1 day. This is blackbox
variant B; the only alpha that reads them (a4) had Sharpe 3.77 vs 3.84 for variant A. The paper engine uses variant A.

## Sizing

Book `paper` holds every basket name (~120–150 after exchange minimums). Share of gross dropped below the $5 minimum:

| Notional per alpha | $2,000 | $5,000 | $10,000 | $20,000 |
|--------------------|--------|--------|---------|---------|
| Skipped gross (median of 8) | 11.7% | 2.9% | 0.6% | 0.1% |

Under ~$5k use `book: "top15"` (15 long + 15 short), which was not backtested per alpha.

## Setup per alpha

1. Create a sub-account, one-way position mode, fund it with USDT (margin needed = notional / leverage, the script
   refuses below 2×).
2. API key with futures trading only, IP whitelisted. Put `BINANCE_KEY_A1` / `BINANCE_SECRET_A1` in `live/.env`.
3. In `config.json` set `enabled: true`, `notional`, `leverage`, `book` for that slot.
4. Dry run `python live/run_live.py --only a1`, then install `com.ml01.live-top8.plist`.

The script refuses duplicated keys and the key of `run_live_single.py`: positions net within one account.

## Files

| Path | Content |
|------|---------|
| `config.json` | slot, key, sign, enabled, notional, leverage, book |
| `state/<slot>.json` | last traded data bar (idempotency) |
| `ledger/<slot>/` | orders, income (funding, commission, realized PnL), account and position snapshots |
| `logs/live.log` | every run |

Telegram alerts (failed orders, errors) go through `~/Desktop/alpha-live/system/tele_report.py` when present.

## Related

- [../README.md](../README.md)
