#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research semi-left deep rebound signals using cached daily bars.

Definition is intentionally simple to reduce overfitting:
- Stock is at least 30% below its prior 120-day high.
- Signal day has either a long lower shadow or a bullish body >= 4%.
- Buy at next trading day's open. No intraday/minute data is used.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


EPS = 1e-12
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
DEFAULT_CACHE = Path("/app/workspace/data_cache/market_data_main_board_2020-01-01_2025-12-31_hfq_200_c9e24d15a649.pkl")
DEFAULT_OUTPUT_DIR = Path("/app/tdx_data/semileft_deep_rebound_research")


def normalize_symbol(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def is_main_board(symbol: str) -> bool:
    return normalize_symbol(symbol).startswith(MAIN_BOARD_PREFIXES)


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def normalize_daily_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    data.columns = [str(column).lower() for column in data.columns]
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "amount" not in data.columns:
        data["amount"] = np.nan
    data = data.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return data


def add_indicators(data: pd.DataFrame, peak_window: int) -> pd.DataFrame:
    output = data.copy()
    output["prev_close"] = output["close"].shift(1)
    output["daily_gain"] = output["close"] / output["prev_close"] - 1.0
    output["body_pct"] = (output["close"] - output["open"]) / output["open"]
    output["abs_body_pct"] = (output["close"] - output["open"]).abs() / output["open"]
    output["lower_shadow_pct"] = (output[["open", "close"]].min(axis=1) - output["low"]) / output["prev_close"]
    output["upper_shadow_pct"] = (output["high"] - output[["open", "close"]].max(axis=1)) / output["prev_close"]
    output["intraday_rebound_pct"] = output["close"] / output["low"] - 1.0
    output["volume_ma20"] = output["volume"].shift(1).rolling(20, min_periods=20).mean()
    output["amount_ma20"] = output["amount"].shift(1).rolling(20, min_periods=20).mean()
    output["volume_ratio_20"] = output["volume"] / output["volume_ma20"]
    output["amount_ratio_20"] = output["amount"] / output["amount_ma20"]
    output["ma20"] = output["close"].rolling(20, min_periods=20).mean()
    output["ma60"] = output["close"].rolling(60, min_periods=60).mean()
    output["ma120"] = output["close"].rolling(120, min_periods=120).mean()
    output["ma60_bias"] = output["close"] / output["ma60"] - 1.0
    output["pre20_return"] = output["close"] / output["close"].shift(20) - 1.0
    output["pre20_down_ratio"] = (output["close"] < output["prev_close"]).shift(1).rolling(20, min_periods=10).mean()
    output["peak_high"] = output["high"].shift(1).rolling(peak_window, min_periods=60).max()
    output["drawdown_from_peak"] = output["close"] / output["peak_high"] - 1.0
    return output


def load_market_data(cache_path: Path, peak_window: int) -> Dict[str, pd.DataFrame]:
    with cache_path.open("rb") as file:
        raw_market_data = pickle.load(file)

    market_data: Dict[str, pd.DataFrame] = {}
    for raw_symbol, raw_data in raw_market_data.items():
        symbol = normalize_symbol(raw_symbol)
        if not is_main_board(symbol) or raw_data is None or raw_data.empty:
            continue
        data = normalize_daily_frame(raw_data)
        if len(data) < peak_window + 80:
            continue
        market_data[symbol] = add_indicators(data, peak_window)
    return market_data


def find_prior_peak_index(data: pd.DataFrame, signal_index: int, peak_window: int) -> Optional[int]:
    start_index = max(0, signal_index - peak_window)
    peak_window_data = data.iloc[start_index:signal_index]
    if peak_window_data.empty:
        return None
    highs = peak_window_data["high"].to_numpy(dtype=float)
    if len(highs) == 0 or np.isnan(highs).all():
        return None
    return int(start_index + np.nanargmax(highs))


def first_30pct_breach_index(data: pd.DataFrame, peak_index: int, signal_index: int, peak_high: float) -> Optional[int]:
    if peak_high <= EPS:
        return None
    close_values = data.iloc[peak_index + 1 : signal_index + 1]["close"].to_numpy(dtype=float)
    hit_offsets = np.flatnonzero(close_values <= peak_high * 0.70 + EPS)
    if len(hit_offsets) == 0:
        return None
    return int(peak_index + 1 + hit_offsets[0])


def forward_return(data: pd.DataFrame, buy_index: int, buy_price: float, days: int) -> float:
    target_index = min(len(data) - 1, buy_index + days - 1)
    return float(data.iloc[target_index]["close"]) / buy_price - 1.0


def hit_target_days(data: pd.DataFrame, buy_index: int, buy_price: float, max_days: int, target: float) -> Optional[int]:
    window = data.iloc[buy_index : min(len(data), buy_index + max_days)]
    for offset, (_, row) in enumerate(window.iterrows(), start=1):
        if float(row["high"]) >= buy_price * (1.0 + target):
            return offset
    return None


def evaluate_signal(
    symbol: str,
    data: pd.DataFrame,
    signal_index: int,
    peak_index: int,
    first_breach_index: Optional[int],
) -> Optional[dict]:
    buy_index = signal_index + 1
    if buy_index >= len(data):
        return None
    forward_window = data.iloc[buy_index : min(len(data), buy_index + 61)]
    if len(forward_window) < 20:
        return None

    signal_row = data.iloc[signal_index]
    buy_row = data.iloc[buy_index]
    peak_row = data.iloc[peak_index]
    buy_price = float(buy_row["open"])
    peak_high = float(peak_row["high"])
    if buy_price <= EPS or peak_high <= EPS:
        return None

    lowest_close = float(data.iloc[peak_index + 1 : signal_index + 1]["close"].min())
    mfe_20 = float(forward_window.head(20)["high"].max()) / buy_price - 1.0
    mae_20 = float(forward_window.head(20)["low"].min()) / buy_price - 1.0
    mfe_60 = float(forward_window["high"].max()) / buy_price - 1.0
    mae_60 = float(forward_window["low"].min()) / buy_price - 1.0
    days_from_peak = signal_index - peak_index
    days_to_30pct = first_breach_index - peak_index if first_breach_index is not None else np.nan

    long_lower_shadow = bool(signal_row["lower_shadow_pct"] >= 0.04 and signal_row["intraday_rebound_pct"] >= 0.04)
    big_bull_body = bool(signal_row["body_pct"] >= 0.04)

    return {
        "symbol": symbol,
        "peak_date": pd.Timestamp(peak_row["date"]).date().isoformat(),
        "signal_date": pd.Timestamp(signal_row["date"]).date().isoformat(),
        "buy_date": pd.Timestamp(buy_row["date"]).date().isoformat(),
        "peak_high": peak_high,
        "signal_open": float(signal_row["open"]),
        "signal_high": float(signal_row["high"]),
        "signal_low": float(signal_row["low"]),
        "signal_close": float(signal_row["close"]),
        "buy_price": buy_price,
        "signal_type": "both" if long_lower_shadow and big_bull_body else ("long_lower_shadow" if long_lower_shadow else "big_bull_body"),
        "long_lower_shadow": long_lower_shadow,
        "big_bull_body": big_bull_body,
        "drawdown_from_peak": float(signal_row["drawdown_from_peak"]),
        "max_close_drawdown_from_peak": lowest_close / peak_high - 1.0,
        "days_from_peak_to_signal": int(days_from_peak),
        "days_from_peak_to_30pct": float(days_to_30pct) if pd.notna(days_to_30pct) else np.nan,
        "drop_speed_to_signal": abs(float(signal_row["drawdown_from_peak"])) / max(days_from_peak, 1),
        "signal_daily_gain": float(signal_row["daily_gain"]),
        "signal_body_pct": float(signal_row["body_pct"]),
        "signal_lower_shadow_pct": float(signal_row["lower_shadow_pct"]),
        "signal_intraday_rebound_pct": float(signal_row["intraday_rebound_pct"]),
        "signal_volume_ratio_20": float(signal_row["volume_ratio_20"]),
        "signal_amount_ratio_20": float(signal_row["amount_ratio_20"]),
        "signal_ma60_bias": float(signal_row["ma60_bias"]) if pd.notna(signal_row["ma60_bias"]) else np.nan,
        "pre20_return": float(signal_row["pre20_return"]) if pd.notna(signal_row["pre20_return"]) else np.nan,
        "pre20_down_ratio": float(signal_row["pre20_down_ratio"]) if pd.notna(signal_row["pre20_down_ratio"]) else np.nan,
        "ret_5d": forward_return(data, buy_index, buy_price, 5),
        "ret_10d": forward_return(data, buy_index, buy_price, 10),
        "ret_20d": forward_return(data, buy_index, buy_price, 20),
        "ret_60d": forward_return(data, buy_index, buy_price, 60),
        "mfe_20d": mfe_20,
        "mae_20d": mae_20,
        "mfe_60d": mfe_60,
        "mae_60d": mae_60,
        "hit_5pct_20d": mfe_20 >= 0.05,
        "hit_10pct_20d": mfe_20 >= 0.10,
        "hit_10pct_60d": mfe_60 >= 0.10,
        "hit_20pct_60d": mfe_60 >= 0.20,
        "loss_8pct_20d": mae_20 <= -0.08,
        "days_to_5pct_20d": hit_target_days(data, buy_index, buy_price, 20, 0.05),
        "days_to_10pct_60d": hit_target_days(data, buy_index, buy_price, 60, 0.10),
    }


def collect_signals(
    market_data: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    peak_window: int,
    min_drawdown: float,
    signal_gap_days: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for symbol, data in market_data.items():
        dates = pd.to_datetime(data["date"]).dt.normalize()
        long_lower_shadow = (data["lower_shadow_pct"] >= 0.04) & (data["intraday_rebound_pct"] >= 0.04)
        big_bull_body = data["body_pct"] >= 0.04
        signal_mask = (
            (dates >= start_date)
            & (dates <= end_date)
            & (data["drawdown_from_peak"] <= -min_drawdown)
            & (long_lower_shadow | big_bull_body)
            & data["peak_high"].notna()
        )
        signal_indices = np.flatnonzero(signal_mask.to_numpy())
        last_signal_index = -10_000
        for signal_index in signal_indices:
            if signal_index - last_signal_index < signal_gap_days:
                continue
            if signal_index >= len(data) - 21:
                continue
            peak_index = find_prior_peak_index(data, signal_index, peak_window)
            if peak_index is None:
                continue
            peak_high = float(data.iloc[peak_index]["high"])
            first_breach = first_30pct_breach_index(data, peak_index, signal_index, peak_high)
            row = evaluate_signal(symbol, data, signal_index, peak_index, first_breach)
            if row is None:
                continue
            rows.append(row)
            last_signal_index = signal_index
    return pd.DataFrame(rows)


def summarize_returns(data: pd.DataFrame, return_column: str) -> dict:
    returns = pd.to_numeric(data[return_column], errors="coerce").dropna()
    if returns.empty:
        return {
            "sample_count": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_loss = -float(losses.sum())
    return {
        "sample_count": int(len(returns)),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum()) / gross_loss if gross_loss > 0 else 0.0,
    }


def bucket_summary(data: pd.DataFrame, column: str, bins: List[float], labels: List[str], return_column: str) -> pd.DataFrame:
    output = data.copy()
    output["bucket"] = pd.cut(output[column], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for bucket, group in output.groupby("bucket", observed=False):
        if group.empty:
            continue
        row = {"feature": column, "bucket": str(bucket)}
        row.update(summarize_returns(group, return_column))
        row["hit_5pct_20d"] = float(group["hit_5pct_20d"].mean())
        row["hit_10pct_20d"] = float(group["hit_10pct_20d"].mean())
        row["hit_20pct_60d"] = float(group["hit_20pct_60d"].mean())
        row["loss_8pct_20d"] = float(group["loss_8pct_20d"].mean())
        row["median_days_to_5pct"] = float(group["days_to_5pct_20d"].dropna().median()) if group["days_to_5pct_20d"].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def render_bucket_table(title: str, table: pd.DataFrame) -> List[str]:
    lines = [title, "-" * 150]
    if table.empty:
        lines.append("无样本")
        return lines
    for _, row in table.iterrows():
        lines.append(
            f"{row['feature']:<28} {row['bucket']:<18} "
            f"n={int(row['sample_count']):>5} "
            f"win={pct(row['win_rate']):>8} "
            f"avg={pct(row['avg_return']):>8} "
            f"med={pct(row['median_return']):>8} "
            f"PF={row['profit_factor']:>5.2f} "
            f"hit5_20={pct(row['hit_5pct_20d']):>8} "
            f"hit10_20={pct(row['hit_10pct_20d']):>8} "
            f"hit20_60={pct(row['hit_20pct_60d']):>8} "
            f"loss8_20={pct(row['loss_8pct_20d']):>8} "
            f"med_days_5={row['median_days_to_5pct']:>4.1f}"
        )
    return lines


def combo_summary(data: pd.DataFrame, return_column: str) -> pd.DataFrame:
    combos = [
        ("shadow_only", data["long_lower_shadow"] & ~data["big_bull_body"]),
        ("big_body_only", data["big_bull_body"] & ~data["long_lower_shadow"]),
        ("both_shadow_and_body", data["long_lower_shadow"] & data["big_bull_body"]),
        ("fast_drop_30d_shadow_or_body", data["days_from_peak_to_30pct"] <= 30),
        ("fast_drop_30d_big_body", (data["days_from_peak_to_30pct"] <= 30) & data["big_bull_body"]),
        ("fast_drop_30d_shadow", (data["days_from_peak_to_30pct"] <= 30) & data["long_lower_shadow"]),
        ("deep35_fast60_confirm", (data["drawdown_from_peak"] <= -0.35) & (data["days_from_peak_to_signal"] <= 60)),
        ("deep35_big_body_vol_ge1", (data["drawdown_from_peak"] <= -0.35) & data["big_bull_body"] & (data["signal_volume_ratio_20"] >= 1.0)),
        ("deep35_shadow_vol_lt1", (data["drawdown_from_peak"] <= -0.35) & data["long_lower_shadow"] & (data["signal_volume_ratio_20"] < 1.0)),
    ]
    rows = []
    for name, mask in combos:
        group = data[mask].copy()
        row = {"combo": name}
        row.update(summarize_returns(group, return_column) if not group.empty else summarize_returns(data.iloc[0:0], return_column))
        row["hit_5pct_20d"] = float(group["hit_5pct_20d"].mean()) if not group.empty else 0.0
        row["hit_10pct_20d"] = float(group["hit_10pct_20d"].mean()) if not group.empty else 0.0
        row["hit_20pct_60d"] = float(group["hit_20pct_60d"].mean()) if not group.empty else 0.0
        row["loss_8pct_20d"] = float(group["loss_8pct_20d"].mean()) if not group.empty else 0.0
        row["avg_mfe_20d"] = float(group["mfe_20d"].mean()) if not group.empty else 0.0
        row["avg_mae_20d"] = float(group["mae_20d"].mean()) if not group.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["avg_return", "win_rate", "sample_count"], ascending=[False, False, False])


def render_combo_table(table: pd.DataFrame) -> List[str]:
    lines = ["组合观察", "-" * 150]
    for _, row in table.iterrows():
        lines.append(
            f"{row['combo']:<32} "
            f"n={int(row['sample_count']):>5} "
            f"win={pct(row['win_rate']):>8} "
            f"avg={pct(row['avg_return']):>8} "
            f"med={pct(row['median_return']):>8} "
            f"PF={row['profit_factor']:>5.2f} "
            f"hit5_20={pct(row['hit_5pct_20d']):>8} "
            f"hit10_20={pct(row['hit_10pct_20d']):>8} "
            f"hit20_60={pct(row['hit_20pct_60d']):>8} "
            f"loss8_20={pct(row['loss_8pct_20d']):>8} "
            f"mfe20={pct(row['avg_mfe_20d']):>8} "
            f"mae20={pct(row['avg_mae_20d']):>8}"
        )
    return lines


def build_report(data: pd.DataFrame, return_column: str, args: argparse.Namespace) -> str:
    lines = [
        "半左侧深跌反弹研究：跌超30%后的长下影/大阳确认",
        "=" * 150,
        f"区间: {args.start} ~ {args.end}",
        f"缓存: {args.market_cache}",
        f"近高窗口: {args.peak_window}日 | 最小回撤: {args.min_drawdown * 100:.0f}% | 同股信号间隔: {args.signal_gap_days}日",
        f"主收益口径: {return_column} > 0，买入价=确认日次日开盘",
        "",
    ]
    overall = summarize_returns(data, return_column)
    lines.append(
        f"全样本 n={overall['sample_count']} win={pct(overall['win_rate'])} "
        f"avg={pct(overall['avg_return'])} med={pct(overall['median_return'])} "
        f"avg_win={pct(overall['avg_win'])} avg_loss={pct(overall['avg_loss'])} PF={overall['profit_factor']:.2f}"
    )
    lines.append(
        f"20日触及+5%={pct(data['hit_5pct_20d'].mean())} | "
        f"20日触及+10%={pct(data['hit_10pct_20d'].mean())} | "
        f"60日触及+20%={pct(data['hit_20pct_60d'].mean())} | "
        f"20日触及-8%={pct(data['loss_8pct_20d'].mean())}"
    )
    lines.append("")

    tables = [
        bucket_summary(data, "signal_type", [-1, 0, 1], ["unused"], return_column) if False else pd.DataFrame(),
        bucket_summary(data, "days_from_peak_to_30pct", [-1, 20, 40, 80, 160, 10_000], ["<=20", "21-40", "41-80", "81-160", ">160"], return_column),
        bucket_summary(data, "days_from_peak_to_signal", [-1, 20, 40, 80, 160, 10_000], ["<=20", "21-40", "41-80", "81-160", ">160"], return_column),
        bucket_summary(data, "drawdown_from_peak", [-10, -0.50, -0.40, -0.35, -0.30], ["<-50%", "-50~-40%", "-40~-35%", "-35~-30%"], return_column),
        bucket_summary(data, "signal_body_pct", [-10, -0.02, 0.0, 0.04, 0.07, 10], ["阴实体<-2%", "-2%~0", "0~4%", "4~7%", ">=7%"], return_column),
        bucket_summary(data, "signal_lower_shadow_pct", [-10, 0.02, 0.04, 0.07, 0.10, 10], ["<2%", "2~4%", "4~7%", "7~10%", ">=10%"], return_column),
        bucket_summary(data, "signal_volume_ratio_20", [-10, 0.6, 1.0, 1.5, 2.5, 10_000], ["<0.6", "0.6~1.0", "1.0~1.5", "1.5~2.5", ">=2.5"], return_column),
        bucket_summary(data, "pre20_return", [-10, -0.35, -0.25, -0.15, -0.05, 10], ["<-35%", "-35~-25%", "-25~-15%", "-15~-5%", ">=-5%"], return_column),
        bucket_summary(data, "pre20_down_ratio", [-1, 0.45, 0.55, 0.65, 0.75, 2], ["<45%", "45~55%", "55~65%", "65~75%", ">=75%"], return_column),
    ]
    titles = [
        "",
        "分桶：从高点到首次跌破-30%的速度",
        "分桶：从高点到确认信号的速度",
        "分桶：信号日距高点跌幅",
        "分桶：确认日实体强度",
        "分桶：确认日下影线强度",
        "分桶：确认日量比",
        "分桶：确认日前20日跌幅",
        "分桶：确认日前20日阴线密度",
    ]
    for title, table in zip(titles, tables):
        if table.empty:
            continue
        lines.extend(render_bucket_table(title, table))
        lines.append("")

    lines.extend(render_combo_table(combo_summary(data, return_column)))
    lines.append("")

    winners = data[data[return_column] > 0]
    losers = data[data[return_column] <= 0]
    compare_columns = [
        "days_from_peak_to_30pct",
        "days_from_peak_to_signal",
        "drawdown_from_peak",
        "signal_body_pct",
        "signal_lower_shadow_pct",
        "signal_volume_ratio_20",
        "pre20_return",
        "pre20_down_ratio",
        "mfe_20d",
        "mae_20d",
    ]
    lines.extend(["盈利单 vs 亏损单中位数", "-" * 100])
    for column in compare_columns:
        if "days" in column:
            lines.append(f"{column:<30} win_med={winners[column].median():.0f}  loss_med={losers[column].median():.0f}")
        else:
            lines.append(f"{column:<30} win_med={pct(winners[column].median())}  loss_med={pct(losers[column].median())}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--peak-window", type=int, default=120)
    parser.add_argument("--min-drawdown", type=float, default=0.30)
    parser.add_argument("--signal-gap-days", type=int, default=20)
    parser.add_argument("--return-column", default="ret_20d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载本地行情缓存...", flush=True)
    market_data = load_market_data(Path(args.market_cache), args.peak_window)
    print(f"有效主板标的: {len(market_data)}", flush=True)
    data = collect_signals(
        market_data,
        pd.Timestamp(args.start),
        pd.Timestamp(args.end),
        args.peak_window,
        args.min_drawdown,
        args.signal_gap_days,
    )
    if data.empty:
        raise ValueError("没有找到满足条件的半左侧深跌反弹样本")

    data.to_csv(output_dir / "semileft_deep_rebound_signals.csv", index=False, encoding="utf-8-sig")
    report = build_report(data, args.return_column, args)
    (output_dir / "semileft_deep_rebound_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
