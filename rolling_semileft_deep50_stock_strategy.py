# -*- coding: utf-8 -*-
"""Rolling-window backtest for the stock-only deep50 enhancement sleeve."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from backtest_semileft_deep_rebound_portfolio import load_market_cache
from semileft_deep50_stock_strategy import (
    DEFAULT_FACTOR_CACHE_DIR,
    DEFAULT_MARKET_CACHE,
    DEFAULT_SIGNAL_CSV,
    DEFAULT_STOCK_LIST_CSV,
    StockDeep50Backtest,
    StockDeep50Config,
    ensure_deep50_signals,
    ensure_mootdx_market_cache,
    equity_stats,
    pct,
)


DEFAULT_ROLLING_OUTPUT_DIR = Path("/app/tdx_data/rolling_semileft_deep50_stock_strategy")


def build_master_calendar(market_data: dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
    dates = set()
    for data in market_data.values():
        if not data.empty:
            dates.update(data.index.tolist())
    return sorted(date for date in dates if start_date <= date <= end_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-csv", default=str(DEFAULT_SIGNAL_CSV))
    parser.add_argument("--market-cache", default=str(DEFAULT_MARKET_CACHE))
    parser.add_argument("--stock-list-csv", default=str(DEFAULT_STOCK_LIST_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROLLING_OUTPUT_DIR))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--window-years", type=float, default=1.0)
    parser.add_argument("--step-days", type=int, default=2)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--stock-sleeve-ratio", type=float, default=0.30)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-daily-buy", type=int, default=3)
    parser.add_argument("--take-profit-pct", type=float, default=0.08)
    parser.add_argument("--stop-loss-pct", type=float, default=0.10)
    parser.add_argument("--max-holding-days", type=int, default=10)
    parser.add_argument("--tdx-dir", default="/app/tdx_data")
    parser.add_argument("--cache-start", default="2015-01-01")
    parser.add_argument("--cache-end", default="", help="为空时使用 --end")
    parser.add_argument("--cache-max-symbols", type=int, default=0)
    parser.add_argument("--cache-min-rows", type=int, default=120)
    parser.add_argument("--cache-adjust", choices=["raw", "hfq_local", "hfq"], default="hfq_local")
    parser.add_argument("--factor-cache-dir", default=str(DEFAULT_FACTOR_CACHE_DIR))
    parser.add_argument("--allow-online-adjust", action="store_true")
    parser.add_argument("--force-refresh-cache", action="store_true")
    parser.add_argument("--refresh-signals", action="store_true")
    parser.add_argument("--no-auto-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--peak-window", type=int, default=120)
    parser.add_argument("--signal-gap-days", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means run all windows; useful for smoke tests.")
    return parser.parse_args()


def run_rolling(args: argparse.Namespace) -> pd.DataFrame:
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    market_cache_path = Path(args.market_cache)
    cache_end = pd.Timestamp(args.cache_end) if args.cache_end else end_date

    if not args.no_auto_cache:
        market_cache_path = ensure_mootdx_market_cache(
            market_cache_path,
            Path(args.stock_list_csv),
            args.tdx_dir,
            pd.Timestamp(args.cache_start),
            cache_end,
            args.cache_max_symbols,
            args.force_refresh_cache,
            args.cache_min_rows,
            args.cache_adjust,
            args.allow_online_adjust,
            Path(args.factor_cache_dir),
        )
        if args.build_cache_only:
            print(f"缓存构建/更新完成：{market_cache_path}")
            return pd.DataFrame()

    print("加载 deep50 信号与行情缓存...", flush=True)
    signals = ensure_deep50_signals(
        Path(args.signals_csv),
        market_cache_path,
        output_dir,
        start_date,
        end_date,
        args.peak_window,
        args.signal_gap_days,
        args.refresh_signals or args.force_refresh_cache,
    )
    market_data = load_market_cache(market_cache_path)
    calendar = build_master_calendar(market_data, start_date, end_date)
    window_bars = int(args.window_years * 252)

    if len(calendar) <= window_bars:
        raise ValueError("数据周期不足以进行滚动窗口回测")
    if args.step_days <= 0:
        raise ValueError("--step-days must be positive")

    start_indices = list(range(0, len(calendar) - window_bars, args.step_days))
    if args.max_windows > 0:
        start_indices = start_indices[: args.max_windows]

    print(f"计划执行 {len(start_indices)} 个滚动窗口，窗口={args.window_years}年，步长={args.step_days}个交易日", flush=True)
    rows = []
    for idx, start_index in enumerate(start_indices, start=1):
        window_start = calendar[start_index]
        window_end = calendar[start_index + window_bars - 1]
        print(
            f"[{idx}/{len(start_indices)}] {window_start.date()} -> {window_end.date()}",
            end="\r",
            flush=True,
        )

        config = StockDeep50Config(
            initial_capital=args.initial_capital,
            stock_sleeve_ratio=args.stock_sleeve_ratio,
            max_positions=args.max_positions,
            max_daily_buy=args.max_daily_buy,
            take_profit_pct=args.take_profit_pct,
            stop_loss_pct=args.stop_loss_pct,
            max_holding_days=args.max_holding_days,
            start_date=window_start,
            end_date=window_end,
        )
        window_calendar = calendar[start_index : start_index + window_bars]
        engine = StockDeep50Backtest(config, signals, market_data, calendar=window_calendar)
        result = engine.run()
        stats = equity_stats(result["equity"], config.initial_capital)
        returns = pd.to_numeric(result["positions"].get("total_pnl_pct", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "start": window_start.date().isoformat(),
                "end": window_end.date().isoformat(),
                "annualized_return": stats.get("annualized_return", 0.0),
                "total_return": stats.get("total_return", 0.0),
                "max_drawdown": stats.get("max_drawdown", 0.0),
                "trade_count": int(len(returns)),
                "trade_win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
                "avg_trade_return": float(returns.mean()) if len(returns) else 0.0,
                "max_position_count": stats.get("max_position_count", 0),
                "avg_capital_utilization": stats.get("avg_capital_utilization", 0.0),
            }
        )

    print()
    return pd.DataFrame(rows)


def build_report(results: pd.DataFrame, args: argparse.Namespace) -> str:
    if results.empty:
        return "无滚动窗口结果"

    profitable = results["annualized_return"] > 0
    worst_by_return = results.sort_values("annualized_return").head(20)
    worst_by_drawdown = results.sort_values("max_drawdown").head(20)

    lines = [
        "deep50 个股增强滚动窗口回测",
        "=" * 100,
        f"区间: {args.start} ~ {args.end}",
        f"窗口: {args.window_years}年，步长: {args.step_days}个交易日，样本数: {len(results)}",
        f"个股增强仓: {args.stock_sleeve_ratio * 100:.0f}%，max_positions={args.max_positions}, max_daily_buy={args.max_daily_buy}",
        f"单票金额: initial_capital * stock_sleeve_ratio / max_positions",
        f"退出: +{args.take_profit_pct * 100:.1f}% 盘中止盈 / -{args.stop_loss_pct * 100:.1f}% 收盘止损 / {args.max_holding_days}日时间止损",
        "-" * 100,
        f"滚动盈利率: {pct(float(profitable.mean()))}",
        f"平均年化收益: {pct(float(results['annualized_return'].mean()))}",
        f"中位年化收益: {pct(float(results['annualized_return'].median()))}",
        f"最差年化收益: {pct(float(results['annualized_return'].min()))}",
        f"最佳年化收益: {pct(float(results['annualized_return'].max()))}",
        f"平均最大回撤: {pct(float(results['max_drawdown'].mean()))}",
        f"最深最大回撤: {pct(float(results['max_drawdown'].min()))}",
        f"平均交易数/窗口: {results['trade_count'].mean():.1f}",
        "",
        "最差收益窗口",
        "-" * 100,
    ]
    for _, row in worst_by_return.iterrows():
        lines.append(
            f"{row['start']} -> {row['end']} | ann={pct(row['annualized_return'])} "
            f"mdd={pct(row['max_drawdown'])} trades={int(row['trade_count'])}"
        )

    lines.extend(["", "最深回撤窗口", "-" * 100])
    for _, row in worst_by_drawdown.iterrows():
        lines.append(
            f"{row['start']} -> {row['end']} | ann={pct(row['annualized_return'])} "
            f"mdd={pct(row['max_drawdown'])} trades={int(row['trade_count'])}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_rolling(args)
    results.to_csv(output_dir / "rolling_results.csv", index=False, encoding="utf-8-sig")
    report = build_report(results, args)
    (output_dir / "rolling_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n已导出: {output_dir}")


if __name__ == "__main__":
    main()
