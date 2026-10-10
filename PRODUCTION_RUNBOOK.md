# Production Strategy Runbook

## Daily workflow

Run from `D:\tdx_api\mootdx`:

```bash
docker-compose exec mootdx python3 /app/workspace/fix_stock_data.py
docker-compose exec mootdx python3 /app/workspace/all_weather_strategy.py
docker-compose exec mootdx python3 /app/workspace/semileft_deep50_stock_strategy.py --start 2026-07-22 --end YYYY-MM-DD --max-daily-buy 3
```

`fix_stock_data.py` updates the four ETF caches only. The deep50 strategy updates its own
market cache from local TDX daily data and local adjustment factors when necessary.

## Production data

Keep these files under `data_cache/`:

- `510880_hfq_cache.csv`
- `511880_hfq_cache.csv`
- `513500_hfq_cache.csv`
- `518880_hfq_cache.csv`
- `market_data_main_board_deep50_2015_hfq_local_mootdx.pkl`
- `market_data_main_board_deep50_2015_hfq_local_mootdx.pkl.meta.txt`

Also keep:

- `D:\通达信` local TDX data
- Docker volume `mootdx_mootdx-config` for local adjustment factors
- `D:\通达信\semileft_deep50_stock_strategy\deep50_signals.csv` and its metadata

## Rolling backtests

```bash
docker-compose exec mootdx python3 /app/workspace/rolling_all_weather.py
```

The deep50 rolling backtest must use a separate signal cache so it cannot overwrite the
live strategy signal cache:

```bash
docker-compose exec mootdx python3 /app/workspace/rolling_semileft_deep50_stock_strategy.py \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --signals-csv /app/tdx_data/rolling_semileft_deep50_stock_strategy/deep50_signals.csv \
  --output-dir /app/tdx_data/rolling_semileft_deep50_stock_strategy
```

Use the same input dates and cache files when comparing results across code changes.
