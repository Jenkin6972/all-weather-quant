# -*- coding: utf-8 -*-
"""Stock-only deep50 semi-left rebound strategy.

This file intentionally keeps the stock sleeve outside all_weather_strategy.py.
It models the stock enhancement sleeve as a fixed percentage of total capital:
position_cash = initial_capital * stock_sleeve_ratio / max_positions.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backtest_semileft_deep_rebound_portfolio import (
    load_market_cache,
    load_signals,
)
from research_semileft_deep_rebound import (
    add_indicators,
    find_prior_peak_index,
    first_30pct_breach_index,
    load_market_data,
    normalize_symbol,
)


EPS = 1e-12
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
DEFAULT_SIGNAL_CSV = Path("/app/tdx_data/semileft_deep50_stock_strategy/deep50_signals.csv")
DEFAULT_MARKET_CACHE = Path("/app/workspace/data_cache/market_data_main_board_deep50_2015_hfq_local_mootdx.pkl")
DEFAULT_STOCK_LIST_CSV = Path("/app/workspace/股票列表/A股列表_20250816_171921.csv")
DEFAULT_FACTOR_CACHE_DIR = Path("/root/.mootdx/caches/factor")
DEFAULT_OUTPUT_DIR = Path("/app/tdx_data/semileft_deep50_stock_strategy")
WINDOW_BARS = 42
SIGNAL_GENERATOR_VERSION = "trade_v1"
SIGNAL_COLUMNS = [
    "symbol",
    "peak_date",
    "signal_date",
    "buy_date",
    "peak_high",
    "signal_open",
    "signal_high",
    "signal_low",
    "signal_close",
    "buy_price",
    "signal_type",
    "long_lower_shadow",
    "big_bull_body",
    "drawdown_from_peak",
    "max_close_drawdown_from_peak",
    "days_from_peak_to_signal",
    "days_from_peak_to_30pct",
    "drop_speed_to_signal",
    "signal_daily_gain",
    "signal_body_pct",
    "signal_lower_shadow_pct",
    "signal_intraday_rebound_pct",
    "signal_volume_ratio_20",
    "signal_amount_ratio_20",
    "signal_ma60_bias",
    "pre20_return",
    "pre20_down_ratio",
    "rank_score",
]


@dataclass
class StockDeep50Config:
    initial_capital: float = 100_000.0
    stock_sleeve_ratio: float = 0.30
    max_positions: int = 10
    max_daily_buy: int = 3
    take_profit_pct: float = 0.08
    stop_loss_pct: float = 0.10
    max_holding_days: int = 10
    add_trigger_pct: float = 0.10
    add_cash_multiplier: float = 1.0
    max_market_down_rate_for_add: float = 0.65
    start_date: pd.Timestamp = pd.Timestamp("2020-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2025-12-31")

    @property
    def stock_sleeve_cash(self) -> float:
        return self.initial_capital * self.stock_sleeve_ratio

    @property
    def position_cash(self) -> float:
        if self.max_positions <= 0:
            return 0.0
        return self.stock_sleeve_cash / self.max_positions


@dataclass
class Position:
    symbol: str
    buy_date: pd.Timestamp
    buy_price: float
    shares: float
    signal_date: pd.Timestamp
    signal_type: str
    rank_score: float
    cost_cash: float = 0.0
    add_cash: float = 0.0
    added: bool = False
    add_date: Optional[pd.Timestamp] = None
    add_price: float = 0.0
    max_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    @property
    def avg_cost(self) -> float:
        if self.shares <= EPS:
            return 0.0
        if self.cost_cash <= EPS:
            return self.buy_price
        return self.cost_cash / self.shares


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def normalize_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    data = signals.copy()
    if data.empty:
        return data
    data["symbol"] = data["symbol"].map(normalize_symbol)
    for column in ["peak_date", "signal_date", "buy_date"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    for column in ["long_lower_shadow", "big_bull_body"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.lower().isin(["true", "1"])
    return data


def is_main_board_symbol(symbol: str) -> bool:
    return normalize_symbol(symbol).startswith(MAIN_BOARD_PREFIXES)


def normalize_daily_cache_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    data.columns = [str(column).lower() for column in data.columns]
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "volume" not in data.columns:
        data["volume"] = np.nan
    if "amount" not in data.columns:
        data["amount"] = np.nan
    data = data.dropna(subset=["date", "open", "high", "low", "close"])
    if data.empty:
        return data
    keep_columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    return data[keep_columns].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def read_stock_list_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    data = pd.read_csv(path, dtype=str)
    code_column = next((column for column in ["code", "symbol", "证券代码", "代码"] if column in data.columns), None)
    if code_column is None:
        code_column = data.columns[0]
    symbols = data[code_column].dropna().map(normalize_symbol).drop_duplicates().tolist()
    return sorted(symbol for symbol in symbols if is_main_board_symbol(symbol))


def read_stock_list_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        symbol = normalize_symbol(text.split()[0].split(",")[0])
        if is_main_board_symbol(symbol):
            symbols.append(symbol)
    return sorted(set(symbols))


def load_main_board_symbols(stock_list_csv: Path) -> Tuple[List[str], str]:
    csv_symbols = read_stock_list_csv(stock_list_csv)
    if csv_symbols:
        return csv_symbols, str(stock_list_csv)

    default_csv_symbols = read_stock_list_csv(DEFAULT_STOCK_LIST_CSV)
    if default_csv_symbols:
        return default_csv_symbols, str(DEFAULT_STOCK_LIST_CSV)

    txt_symbols = read_stock_list_txt(Path("/app/workspace/A股列表.txt"))
    if txt_symbols:
        return txt_symbols, "/app/workspace/A股列表.txt"

    raise FileNotFoundError("未找到可用的主板股票列表 CSV/TXT，请通过 --stock-list-csv 指定")


def validate_local_factor_coverage(symbols: List[str], factor_cache_dir: Path) -> None:
    missing_symbols = []
    for symbol in symbols:
        factor_path = factor_cache_dir / f"{stock_market_prefix(symbol)}{normalize_symbol(symbol)}.plk"
        if not factor_path.exists():
            missing_symbols.append(symbol)

    if not missing_symbols:
        return

    preview = ", ".join(missing_symbols[:20])
    raise FileNotFoundError(
        f"本地复权因子缺失 {len(missing_symbols)} 只，无法保证离线后复权一致性。"
        f"示例：{preview}。"
    )


def stock_market_prefix(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    if symbol.startswith("6"):
        return "sh"
    return "sz"


def load_local_factor(symbol: str, factor_cache_dir: Path) -> pd.DataFrame:
    factor_path = factor_cache_dir / f"{stock_market_prefix(symbol)}{normalize_symbol(symbol)}.plk"
    if not factor_path.exists():
        return pd.DataFrame()
    with factor_path.open("rb") as file:
        factor = pickle.load(file)
    if factor is None or factor.empty or "factor" not in factor.columns:
        return pd.DataFrame()
    output = factor.copy()
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
        output = output.set_index("date")
    else:
        output.index = pd.to_datetime(output.index, errors="coerce").normalize()
    output["factor"] = pd.to_numeric(output["factor"], errors="coerce")
    output = output.dropna(subset=["factor"])
    return output.sort_index()


def apply_local_hfq(raw_data: pd.DataFrame, symbol: str, factor_cache_dir: Path) -> pd.DataFrame:
    data = normalize_daily_cache_frame(raw_data)
    if data.empty:
        return data

    factor = load_local_factor(symbol, factor_cache_dir)
    if factor.empty:
        return data

    indexed = data.set_index("date", drop=False).sort_index()
    factor_slice = factor.loc[indexed.index[0] : indexed.index[-1], ["factor"]]
    indexed = pd.concat([indexed, factor_slice], axis=1)
    indexed["factor"] = indexed["factor"].ffill().fillna(1.0).astype(float)
    for column in ["open", "high", "low", "close"]:
        indexed[column] = indexed[column] * indexed["factor"]
    indexed = indexed.drop(columns=["factor"], errors="ignore")
    return normalize_daily_cache_frame(indexed.reset_index(drop=True))


class MootdxDailyCacheBuilder:
    def __init__(self, tdx_dir: str, adjust: str, allow_online_adjust: bool, factor_cache_dir: Path) -> None:
        from mootdx.reader import Reader

        if adjust == "hfq" and not allow_online_adjust:
            raise ValueError(
                "mootdx 的后复权 adjust='hfq' 会联网获取复权因子。"
                "若坚持生成后复权缓存，请显式增加 --allow-online-adjust；"
                "若要严格离线后复权，请使用默认 --cache-adjust hfq_local。"
            )
        self.reader = Reader.factory(market="std", tdxdir=tdx_dir)
        self.adjust = adjust
        self.factor_cache_dir = factor_cache_dir

    def load_daily(self, symbol: str) -> pd.DataFrame:
        if self.adjust == "hfq":
            raw_data = self.reader.daily(symbol=symbol, adjust="hfq")
        elif self.adjust == "hfq_local":
            raw_data = self.reader.daily(symbol=symbol)
            return apply_local_hfq(raw_data, symbol, self.factor_cache_dir)
        else:
            raw_data = self.reader.daily(symbol=symbol)
        if raw_data is None or raw_data.empty:
            return pd.DataFrame()
        return normalize_daily_cache_frame(raw_data)


def cache_covers_end_date(data: pd.DataFrame, end_date: pd.Timestamp) -> bool:
    if data.empty or "date" not in data.columns:
        return False
    max_date = pd.to_datetime(data["date"], errors="coerce").max()
    return bool(pd.notna(max_date) and max_date >= end_date)


def market_cache_covers_end_date(market_data: Dict[str, pd.DataFrame], end_date: pd.Timestamp) -> bool:
    if not market_data:
        return False
    max_dates = []
    for data in market_data.values():
        if data.empty or "date" not in data.columns:
            continue
        max_date = pd.to_datetime(data["date"], errors="coerce").max()
        if pd.notna(max_date):
            max_dates.append(max_date)
    if not max_dates:
        return False
    return max(max_dates) >= end_date


def load_raw_market_cache(path: Path) -> Dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        raw_market_data = pickle.load(file)
    market_data: Dict[str, pd.DataFrame] = {}
    for raw_symbol, raw_data in raw_market_data.items():
        symbol = normalize_symbol(raw_symbol)
        if raw_data is None or raw_data.empty:
            continue
        data = normalize_daily_cache_frame(raw_data)
        if not data.empty:
            market_data[symbol] = data
    return market_data


def write_market_cache(path: Path, market_data: Dict[str, pd.DataFrame], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(market_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path = path.with_suffix(path.suffix + ".meta.txt")
    lines = [f"{key}: {value}" for key, value in meta.items()]
    meta_path.write_text("\n".join(lines), encoding="utf-8")


def read_simple_meta(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    meta: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def print_market_cache_info(cache_path: Path) -> None:
    market_data = load_raw_market_cache(cache_path)
    if not market_data:
        print(f"缓存为空或不存在：{cache_path}")
        return

    min_dates = []
    max_dates = []
    row_count = 0
    columns: List[str] = []
    for data in market_data.values():
        dates = pd.to_datetime(data["date"], errors="coerce").dropna()
        if dates.empty:
            continue
        min_dates.append(dates.min())
        max_dates.append(dates.max())
        row_count += len(data)
        columns = list(data.columns)

    print("缓存信息")
    print("-" * 80)
    print(f"路径: {cache_path}")
    print(f"标的数: {len(market_data)}")
    print(f"总行数: {row_count}")
    if min_dates and max_dates:
        print(f"最早日期: {min(min_dates).date().isoformat()}")
        print(f"最晚日期: {max(max_dates).date().isoformat()}")
    print(f"字段: {columns}")


def signal_cache_covers_range(
    signals_csv: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    market_cache: Path,
) -> bool:
    meta = read_simple_meta(signals_csv.with_suffix(signals_csv.suffix + ".meta.txt"))
    if meta:
        if meta.get("market_cache") != str(market_cache):
            return False
        try:
            cached_start = pd.Timestamp(meta.get("start_date"))
            cached_end = pd.Timestamp(meta.get("end_date"))
        except Exception:
            return False
        if cached_start > start_date or cached_end < end_date:
            return False
        if meta.get("signal_generator") == SIGNAL_GENERATOR_VERSION:
            return signals_csv.exists()

    try:
        header_data = pd.read_csv(signals_csv, dtype={"symbol": str}, usecols=["signal_date", "buy_date"])
    except Exception:
        return False
    if header_data.empty:
        return False
    signal_dates = pd.to_datetime(header_data["signal_date"], errors="coerce").dropna()
    buy_dates = pd.to_datetime(header_data["buy_date"], errors="coerce").dropna()
    if signal_dates.empty or buy_dates.empty:
        return False
    return signal_dates.min() <= start_date and buy_dates.max() >= end_date


def build_trade_signal_row(
    symbol: str,
    data: pd.DataFrame,
    signal_index: int,
    peak_index: int,
    first_breach_index: Optional[int],
) -> Optional[dict]:
    buy_index = signal_index + 1
    if buy_index >= len(data):
        return None

    signal_row = data.iloc[signal_index]
    buy_row = data.iloc[buy_index]
    peak_row = data.iloc[peak_index]
    buy_price = float(buy_row["open"])
    peak_high = float(peak_row["high"])
    if buy_price <= EPS or peak_high <= EPS:
        return None

    drawdown_from_peak = float(signal_row["drawdown_from_peak"])
    pre20_return = float(signal_row["pre20_return"]) if pd.notna(signal_row["pre20_return"]) else 0.0
    pre20_down_ratio = float(signal_row["pre20_down_ratio"]) if pd.notna(signal_row["pre20_down_ratio"]) else 0.0
    long_lower_shadow = bool(
        float(signal_row["lower_shadow_pct"]) >= 0.04
        and float(signal_row["intraday_rebound_pct"]) >= 0.04
    )
    big_bull_body = bool(float(signal_row["body_pct"]) >= 0.04)
    days_from_peak = signal_index - peak_index
    days_to_30pct = first_breach_index - peak_index if first_breach_index is not None else np.nan
    lowest_close = float(data.iloc[peak_index + 1 : signal_index + 1]["close"].min())
    rank_score = (
        min(max(-pre20_return, 0.0), 0.5) * 4.0
        + pre20_down_ratio * 2.0
        + float(long_lower_shadow)
        + float(big_bull_body)
        + min(max(-drawdown_from_peak, 0.0), 0.8)
    )

    return {
        "symbol": symbol,
        "peak_date": pd.Timestamp(peak_row["date"]).normalize(),
        "signal_date": pd.Timestamp(signal_row["date"]).normalize(),
        "buy_date": pd.Timestamp(buy_row["date"]).normalize(),
        "peak_high": peak_high,
        "signal_open": float(signal_row["open"]),
        "signal_high": float(signal_row["high"]),
        "signal_low": float(signal_row["low"]),
        "signal_close": float(signal_row["close"]),
        "buy_price": buy_price,
        "signal_type": "both" if long_lower_shadow and big_bull_body else ("long_lower_shadow" if long_lower_shadow else "big_bull_body"),
        "long_lower_shadow": long_lower_shadow,
        "big_bull_body": big_bull_body,
        "drawdown_from_peak": drawdown_from_peak,
        "max_close_drawdown_from_peak": lowest_close / peak_high - 1.0,
        "days_from_peak_to_signal": int(days_from_peak),
        "days_from_peak_to_30pct": float(days_to_30pct) if pd.notna(days_to_30pct) else np.nan,
        "drop_speed_to_signal": abs(drawdown_from_peak) / max(days_from_peak, 1),
        "signal_daily_gain": float(signal_row["daily_gain"]) if pd.notna(signal_row["daily_gain"]) else np.nan,
        "signal_body_pct": float(signal_row["body_pct"]) if pd.notna(signal_row["body_pct"]) else np.nan,
        "signal_lower_shadow_pct": float(signal_row["lower_shadow_pct"]) if pd.notna(signal_row["lower_shadow_pct"]) else np.nan,
        "signal_intraday_rebound_pct": float(signal_row["intraday_rebound_pct"]) if pd.notna(signal_row["intraday_rebound_pct"]) else np.nan,
        "signal_volume_ratio_20": float(signal_row["volume_ratio_20"]) if pd.notna(signal_row["volume_ratio_20"]) else np.nan,
        "signal_amount_ratio_20": float(signal_row["amount_ratio_20"]) if pd.notna(signal_row["amount_ratio_20"]) else np.nan,
        "signal_ma60_bias": float(signal_row["ma60_bias"]) if pd.notna(signal_row["ma60_bias"]) else np.nan,
        "pre20_return": pre20_return,
        "pre20_down_ratio": pre20_down_ratio,
        "rank_score": rank_score,
    }


def collect_trade_signals(
    market_data: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    peak_window: int,
    signal_gap_days: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    signal_start = start_date - pd.Timedelta(days=max(signal_gap_days * 3, 10))
    for symbol, data in market_data.items():
        if data.empty:
            continue

        dates = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        long_lower_shadow = (data["lower_shadow_pct"] >= 0.04) & (data["intraday_rebound_pct"] >= 0.04)
        big_bull_body = data["body_pct"] >= 0.04
        signal_mask = (
            (dates >= signal_start)
            & (dates <= end_date)
            & (data["drawdown_from_peak"] <= -0.30)
            & (long_lower_shadow | big_bull_body)
            & data["peak_high"].notna()
        )
        signal_indices = np.flatnonzero(signal_mask.to_numpy())
        for signal_index in signal_indices:
            peak_index = find_prior_peak_index(data, signal_index, peak_window)
            if peak_index is None:
                continue

            peak_high = float(data.iloc[peak_index]["high"])
            first_breach = first_30pct_breach_index(data, peak_index, signal_index, peak_high)
            row = build_trade_signal_row(symbol, data, signal_index, peak_index, first_breach)
            if row is None:
                continue

            if row["buy_date"] < start_date or row["buy_date"] > end_date:
                continue
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS).sort_values(
        ["buy_date", "rank_score", "symbol"],
        ascending=[True, False, True],
    )


def write_signal_cache_meta(
    signals_csv: Path,
    market_cache: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    peak_window: int,
    signal_gap_days: int,
    signal_count: int,
) -> None:
    meta = {
        "signal_generator": SIGNAL_GENERATOR_VERSION,
        "market_cache": str(market_cache),
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "peak_window": peak_window,
        "signal_gap_days": signal_gap_days,
        "signal_count": signal_count,
    }
    lines = [f"{key}: {value}" for key, value in meta.items()]
    signals_csv.with_suffix(signals_csv.suffix + ".meta.txt").write_text("\n".join(lines), encoding="utf-8")


def ensure_mootdx_market_cache(
    cache_path: Path,
    stock_list_csv: Path,
    tdx_dir: str,
    cache_start: pd.Timestamp,
    cache_end: pd.Timestamp,
    max_symbols: int,
    force_refresh: bool,
    min_rows: int,
    adjust: str,
    allow_online_adjust: bool,
    factor_cache_dir: Path,
) -> Path:
    symbols, source = load_main_board_symbols(stock_list_csv)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    if adjust == "hfq_local":
        validate_local_factor_coverage(symbols, factor_cache_dir)

    if cache_path.exists() and not force_refresh:
        market_data = load_raw_market_cache(cache_path)
    else:
        market_data = {}

    need_update = force_refresh or not cache_path.exists()
    if not need_update:
        need_update = not market_cache_covers_end_date(market_data, cache_end)

    if not need_update:
        print(f"发现主板日线缓存，直接使用：{cache_path.name}", flush=True)
        return cache_path

    print(f"开始构建/更新主板日线缓存：{cache_path}", flush=True)
    print(f"股票池来源：{source}，主板标的 {len(symbols)} 只", flush=True)
    if adjust == "hfq":
        print("数据源：mootdx Reader.daily(adjust='hfq')；注意该模式会联网获取复权因子", flush=True)
    elif adjust == "hfq_local":
        print(f"数据源：本地 mootdx Reader.daily(symbol) + 本地复权因子 {factor_cache_dir}，不调用接口", flush=True)
    else:
        print("数据源：仅使用本地 mootdx Reader.daily(symbol)，不调用东方财富/新浪复权接口", flush=True)

    loader = MootdxDailyCacheBuilder(tdx_dir, adjust, allow_online_adjust, factor_cache_dir)
    updated_market_data: Dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        existing_data = market_data.get(symbol, pd.DataFrame())
        if not force_refresh and cache_covers_end_date(existing_data, cache_end):
            updated_market_data[symbol] = existing_data
        else:
            try:
                loaded_data = loader.load_daily(symbol)
            except Exception as exc:
                print(f"[跳过] {symbol} mootdx 日线读取失败：{exc}", flush=True)
                loaded_data = pd.DataFrame()

            if loaded_data.empty:
                if not existing_data.empty:
                    updated_market_data[symbol] = existing_data
            else:
                if not existing_data.empty:
                    loaded_data = pd.concat([existing_data, loaded_data], ignore_index=True)
                    loaded_data = normalize_daily_cache_frame(loaded_data)
                use_data = loaded_data[(loaded_data["date"] >= cache_start) & (loaded_data["date"] <= cache_end)].copy()
                if len(use_data) >= min_rows:
                    updated_market_data[symbol] = use_data

        if index % 50 == 0 or index == total:
            print(f"缓存进度：{index}/{total}，有效标的 {len(updated_market_data)}", flush=True)

    if not updated_market_data:
        raise ValueError("未能构建任何有效主板后复权行情缓存")

    meta = {
        "source": f"mootdx Reader.daily adjust={adjust}",
        "stock_list": source,
        "tdx_dir": tdx_dir,
        "adjust": adjust,
        "allow_online_adjust": allow_online_adjust,
        "factor_cache_dir": factor_cache_dir,
        "cache_start": cache_start.date().isoformat(),
        "cache_end": cache_end.date().isoformat(),
        "symbol_count": len(updated_market_data),
    }
    write_market_cache(cache_path, updated_market_data, meta)
    print(f"主板日线缓存已写入：{cache_path}", flush=True)
    return cache_path


def ensure_deep50_signals(
    signals_csv: Path,
    market_cache: Path,
    output_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    peak_window: int,
    signal_gap_days: int,
    force_refresh: bool,
) -> pd.DataFrame:
    generated_signals = False
    if (
        signals_csv.exists()
        and not force_refresh
        and signal_cache_covers_range(signals_csv, start_date, end_date, market_cache)
    ):
        signals = load_signals(signals_csv)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        signals_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"使用行情缓存生成 deep50 信号：{signals_csv}", flush=True)
        market_data_for_signal = load_market_data(market_cache, peak_window)
        signals = collect_trade_signals(
            market_data_for_signal,
            start_date,
            end_date,
            peak_window,
            signal_gap_days,
        )
        generated_signals = True

    signals = normalize_signal_frame(signals)
    signals = signals[signals["drawdown_from_peak"] <= -0.50].copy()
    if signals.empty:
        signals = pd.DataFrame(columns=SIGNAL_COLUMNS)
        if generated_signals:
            signals.to_csv(signals_csv, index=False, encoding="utf-8-sig")
            write_signal_cache_meta(
                signals_csv,
                market_cache,
                start_date,
                end_date,
                peak_window,
                signal_gap_days,
                0,
            )
        return signals

    signals["rank_score"] = (
        (-signals["pre20_return"].fillna(0.0)).clip(0, 0.5) * 4.0
        + signals["pre20_down_ratio"].fillna(0.0) * 2.0
        + signals["long_lower_shadow"].astype(float) * 1.0
        + signals["big_bull_body"].astype(float) * 1.0
        + (-signals["drawdown_from_peak"].fillna(0.0)).clip(0, 0.8)
    )
    output = signals.sort_values(["buy_date", "rank_score", "symbol"], ascending=[True, False, True])
    if generated_signals:
        output.to_csv(signals_csv, index=False, encoding="utf-8-sig")
        write_signal_cache_meta(
            signals_csv,
            market_cache,
            start_date,
            end_date,
            peak_window,
            signal_gap_days,
            len(output),
        )
    return output


def build_calendar(market_data: Dict[str, pd.DataFrame], start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
    dates = set()
    for data in market_data.values():
        if not data.empty:
            dates.update(data.index.tolist())
    return sorted(date for date in dates if start_date <= date <= end_date)


def compute_market_down_rate(market_data: Dict[str, pd.DataFrame]) -> pd.Series:
    frames = []
    for data in market_data.values():
        if data.empty or "prev_close" not in data.columns:
            continue

        frame = data[["date", "close", "prev_close"]].copy()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["prev_close"] = pd.to_numeric(frame["prev_close"], errors="coerce")
        frame = frame[(frame["close"] > EPS) & (frame["prev_close"] > EPS)]
        if frame.empty:
            continue

        frame["is_down"] = frame["close"] < frame["prev_close"]
        frames.append(frame[["date", "is_down"]])

    if not frames:
        return pd.Series(dtype=float)

    all_rows = pd.concat(frames, ignore_index=True)
    grouped = all_rows.groupby("date")["is_down"]
    return grouped.sum() / grouped.count()


class StockDeep50Backtest:
    def __init__(
        self,
        config: StockDeep50Config,
        signals: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        calendar: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        self.config = config
        self.signals = signals[
            (signals["buy_date"] >= config.start_date)
            & (signals["buy_date"] <= config.end_date)
        ].copy()
        self.signals_by_date = {
            date: group.sort_values(["rank_score", "symbol"], ascending=[False, True]).copy()
            for date, group in self.signals.groupby("buy_date")
        }
        self.market_data = market_data
        self.market_down_rate = compute_market_down_rate(market_data)
        if calendar is None:
            self.calendar = build_calendar(market_data, config.start_date, config.end_date)
        else:
            self.calendar = [date for date in calendar if config.start_date <= date <= config.end_date]
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_sells: Dict[str, str] = {}
        self.pending_adds: Dict[str, str] = {}
        self.sold_today: Set[str] = set()
        self.equity_rows: List[dict] = []
        self.position_rows: List[dict] = []
        self.trade_rows: List[dict] = []
        self.skip_rows: List[dict] = []

    def open_position_rows(self) -> List[dict]:
        rows = []
        for position in self.positions.values():
            rows.append(
                {
                    "symbol": position.symbol,
                    "buy_date": position.buy_date,
                    "buy_price": position.buy_price,
                    "avg_cost": position.avg_cost,
                    "shares": position.shares,
                    "cost_cash": position.cost_cash,
                    "add_cash": position.add_cash,
                    "signal_date": position.signal_date,
                    "signal_type": position.signal_type,
                    "rank_score": position.rank_score,
                    "added": position.added,
                    "add_date": position.add_date,
                    "add_price": position.add_price,
                    "max_profit_pct": position.max_profit_pct,
                    "max_drawdown_pct": position.max_drawdown_pct,
                }
            )
        return rows

    def get_row(self, symbol: str, date: pd.Timestamp) -> Optional[pd.Series]:
        data = self.market_data.get(symbol)
        if data is None or date not in data.index:
            return None
        return data.loc[date]

    def is_limit_down_open(self, row: pd.Series) -> bool:
        prev_close = float(row["prev_close"]) if pd.notna(row.get("prev_close")) else 0.0
        open_price = float(row["open"]) if pd.notna(row.get("open")) else 0.0
        return prev_close > EPS and open_price > EPS and open_price / prev_close - 1.0 <= -0.095 + EPS

    def holding_days(self, position: Position, date: pd.Timestamp) -> int:
        data = self.market_data.get(position.symbol)
        if data is None:
            return 0
        return int(((data.index >= position.buy_date) & (data.index <= date)).sum())

    def sell_position(self, date: pd.Timestamp, symbol: str, price: float, reason: str) -> None:
        position = self.positions.get(symbol)
        if position is None or price <= EPS:
            return

        cost_cash = position.cost_cash if position.cost_cash > EPS else position.shares * position.buy_price
        proceeds = position.shares * price
        pnl_cash = proceeds - cost_cash
        pnl = pnl_cash / cost_cash if cost_cash > EPS else 0.0
        self.cash += proceeds
        self.position_rows.append(
            {
                "symbol": symbol,
                "signal_date": position.signal_date,
                "buy_date": position.buy_date,
                "sell_date": date,
                "buy_price": position.buy_price,
                "avg_cost": position.avg_cost,
                "sell_price": price,
                "shares": position.shares,
                "cost_cash": cost_cash,
                "total_pnl_pct": pnl,
                "total_pnl_cash": pnl_cash,
                "holding_days": self.holding_days(position, date),
                "reason": reason,
                "signal_type": position.signal_type,
                "rank_score": position.rank_score,
                "add_trigger_pct": self.config.add_trigger_pct,
                "added": position.added,
                "add_date": position.add_date,
                "add_price": position.add_price,
                "max_profit_pct": position.max_profit_pct,
                "max_drawdown_pct": position.max_drawdown_pct,
            }
        )
        self.trade_rows.append(
            {
                "date": date,
                "symbol": symbol,
                "action": "SELL",
                "price": price,
                "reason": reason,
                "pnl_pct": pnl,
                "pnl_cash": pnl_cash,
            }
        )
        self.pending_adds.pop(symbol, None)
        self.sold_today.add(symbol)
        del self.positions[symbol]

    def execute_pending_sells(self, date: pd.Timestamp) -> None:
        next_pending: Dict[str, str] = {}
        for symbol, reason in self.pending_sells.items():
            row = self.get_row(symbol, date)
            if row is None:
                next_pending[symbol] = reason
                continue
            if self.is_limit_down_open(row):
                next_pending[symbol] = reason
                self.skip_rows.append({"date": date, "symbol": symbol, "reason": "limit_down_open_sell_roll"})
                continue
            self.sell_position(date, symbol, float(row["open"]), reason)
        self.pending_sells = next_pending

    def execute_average_down(self, date: pd.Timestamp, position: Position, price: float, reason: str) -> None:
        if price <= EPS:
            return

        cash_to_use = min(position.add_cash, self.cash)
        if cash_to_use <= EPS:
            return

        added_shares = cash_to_use / price
        position.shares += added_shares
        position.cost_cash += cash_to_use
        position.added = True
        position.add_date = date
        position.add_price = price
        self.cash -= cash_to_use
        self.trade_rows.append(
            {
                "date": date,
                "symbol": position.symbol,
                "action": "BUY",
                "price": price,
                "cash": cash_to_use,
                "reason": reason,
            }
        )

    def execute_pending_adds(self, date: pd.Timestamp) -> None:
        next_pending: Dict[str, str] = {}
        for symbol, reason in self.pending_adds.items():
            position = self.positions.get(symbol)
            if position is None or position.added or symbol in self.pending_sells:
                continue

            row = self.get_row(symbol, date)
            if row is None:
                next_pending[symbol] = reason
                continue

            add_price = float(row["open"])
            if add_price <= EPS:
                next_pending[symbol] = reason
                continue

            self.execute_average_down(date, position, add_price, reason)

        self.pending_adds = next_pending

    def execute_buys(self, date: pd.Timestamp) -> None:
        candidates = self.signals_by_date.get(date)
        if candidates is None or candidates.empty:
            return

        buy_count = 0
        for _, signal in candidates.iterrows():
            if buy_count >= self.config.max_daily_buy:
                break
            if len(self.positions) >= self.config.max_positions:
                break

            symbol = normalize_symbol(signal["symbol"])
            if symbol in self.positions:
                self.skip_rows.append({"date": date, "symbol": symbol, "reason": "already_holding"})
                continue
            if symbol in self.sold_today:
                self.skip_rows.append({"date": date, "symbol": symbol, "reason": "sold_today"})
                continue

            row = self.get_row(symbol, date)
            if row is None:
                self.skip_rows.append({"date": date, "symbol": symbol, "reason": "missing_buy_bar"})
                continue

            open_price = float(row["open"])
            if open_price <= EPS:
                self.skip_rows.append({"date": date, "symbol": symbol, "reason": "bad_open"})
                continue

            cash_to_use = min(self.config.position_cash, self.cash)
            if cash_to_use <= EPS:
                break

            shares = cash_to_use / open_price
            self.cash -= cash_to_use
            self.positions[symbol] = Position(
                symbol=symbol,
                buy_date=date,
                buy_price=open_price,
                shares=shares,
                signal_date=signal["signal_date"],
                signal_type=str(signal["signal_type"]),
                rank_score=float(signal["rank_score"]),
                cost_cash=cash_to_use,
                add_cash=self.config.position_cash * self.config.add_cash_multiplier,
            )
            buy_count += 1
            self.trade_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "action": "BUY",
                    "price": open_price,
                    "cash": cash_to_use,
                    "reason": "entry",
                    "rank_score": float(signal["rank_score"]),
                }
            )

    def average_down_block_reason(self, date: pd.Timestamp) -> str:
        down_rate = self.market_down_rate.get(date, np.nan) if not self.market_down_rate.empty else np.nan
        if pd.notna(down_rate) and float(down_rate) >= self.config.max_market_down_rate_for_add - EPS:
            return "average_down_blocked_market_down_rate"
        return ""

    def maybe_average_down(self, date: pd.Timestamp, position: Position, row: pd.Series) -> None:
        if position.added:
            return
        if position.symbol in self.pending_adds:
            return
        if self.config.add_trigger_pct <= EPS or position.add_cash <= EPS:
            return

        trigger_price = position.buy_price * (1.0 - self.config.add_trigger_pct)
        low_price = float(row["low"])
        if low_price > trigger_price + EPS:
            return

        block_reason = self.average_down_block_reason(date)
        if block_reason:
            down_rate = self.market_down_rate.get(date, np.nan) if not self.market_down_rate.empty else np.nan
            self.skip_rows.append(
                {
                    "date": date,
                    "symbol": position.symbol,
                    "reason": block_reason,
                    "down_rate": down_rate,
                }
            )
            return

        self.pending_adds[position.symbol] = f"average_down_next_open_{self.config.add_trigger_pct:.0%}"

    def evaluate_positions(self, date: pd.Timestamp) -> None:
        for symbol, position in list(self.positions.items()):
            if symbol in self.pending_sells:
                continue

            row = self.get_row(symbol, date)
            if row is None:
                continue

            was_added_before = position.added
            self.maybe_average_down(date, position, row)
            added_today = (not was_added_before) and position.added and position.add_date == date

            open_price = float(row["open"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            close_price = float(row["close"])
            avg_cost = position.avg_cost
            if avg_cost <= EPS:
                continue

            high_return = high_price / avg_cost - 1.0
            low_return = low_price / avg_cost - 1.0
            close_return = close_price / avg_cost - 1.0
            position.max_profit_pct = max(position.max_profit_pct, high_return)
            position.max_drawdown_pct = min(position.max_drawdown_pct, low_return)

            holding_days = self.holding_days(position, date)
            target_price = avg_cost * (1.0 + self.config.take_profit_pct)
            if high_return >= self.config.take_profit_pct:
                if holding_days <= 1:
                    self.pending_sells[symbol] = "tp_buy_day_next_open"
                    self.pending_adds.pop(symbol, None)
                elif added_today:
                    self.pending_sells[symbol] = "tp_add_day_next_open"
                    self.pending_adds.pop(symbol, None)
                else:
                    sell_price = open_price if open_price >= target_price else target_price
                    self.sell_position(date, symbol, sell_price, "take_profit")
                continue

            if close_return <= -self.config.stop_loss_pct:
                self.pending_sells[symbol] = "stop_loss"
                self.pending_adds.pop(symbol, None)
                continue

            if holding_days >= self.config.max_holding_days:
                self.pending_sells[symbol] = "time_stop"
                self.pending_adds.pop(symbol, None)

    def mark_to_market(self, date: pd.Timestamp) -> None:
        stock_value = 0.0
        for symbol, position in self.positions.items():
            row = self.get_row(symbol, date)
            price = float(row["close"]) if row is not None else position.avg_cost
            stock_value += position.shares * price

        total_value = self.cash + stock_value
        self.equity_rows.append(
            {
                "date": date,
                "total_value": total_value,
                "cash": self.cash,
                "stock_value": stock_value,
                "position_count": len(self.positions),
                "capital_utilization": stock_value / total_value if total_value > EPS else 0.0,
            }
        )

    def close_end_positions(self) -> None:
        if not self.calendar:
            return
        final_date = self.calendar[-1]
        for symbol in list(self.positions.keys()):
            row = self.get_row(symbol, final_date)
            if row is None:
                continue
            self.sell_position(final_date, symbol, float(row["close"]), "end_mark_to_market")

    def run(self, close_end_positions: bool = True) -> dict:
        for date in self.calendar:
            self.sold_today = set()
            self.execute_pending_sells(date)
            self.execute_pending_adds(date)
            self.execute_buys(date)
            self.evaluate_positions(date)
            self.mark_to_market(date)

        last_date = self.calendar[-1] if self.calendar else pd.NaT
        open_positions_before_close = self.open_position_rows()
        pending_sells_before_close = dict(self.pending_sells)
        pending_adds_before_close = dict(self.pending_adds)

        if close_end_positions:
            self.close_end_positions()

        return {
            "equity": pd.DataFrame(self.equity_rows),
            "positions": pd.DataFrame(self.position_rows),
            "trades": pd.DataFrame(self.trade_rows),
            "skips": pd.DataFrame(self.skip_rows),
            "open_positions": pd.DataFrame(open_positions_before_close),
            "pending_sells": pending_sells_before_close,
            "pending_adds": pending_adds_before_close,
            "last_date": last_date,
        }


def equity_stats(equity: pd.DataFrame, initial_capital: float) -> dict:
    if equity.empty:
        return {}

    values = pd.to_numeric(equity["total_value"], errors="coerce").dropna().reset_index(drop=True)
    if values.empty:
        return {}

    total_return = float(values.iloc[-1]) / initial_capital - 1.0
    drawdown = values / values.cummax() - 1.0
    window_return = (values.shift(-WINDOW_BARS) / values - 1.0).dropna()
    return {
        "total_return": total_return,
        "annualized_return": (1.0 + total_return) ** (252.0 / len(values)) - 1.0,
        "max_drawdown": float(drawdown.min()),
        "win_42": float((window_return > 0).mean()) if not window_return.empty else np.nan,
        "avg_42": float(window_return.mean()) if not window_return.empty else np.nan,
        "median_42": float(window_return.median()) if not window_return.empty else np.nan,
        "min_42": float(window_return.min()) if not window_return.empty else np.nan,
        "avg_capital_utilization": float(equity["capital_utilization"].mean()),
        "max_position_count": int(equity["position_count"].max()),
    }


def position_stats(positions: pd.DataFrame) -> dict:
    if positions.empty:
        return {
            "trade_count": 0,
            "trade_win_rate": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
            "add_count": 0,
            "add_rate": 0.0,
            "avg_holding_days": 0.0,
            "profit_factor": 0.0,
        }

    returns = pd.to_numeric(positions["total_pnl_pct"], errors="coerce").dropna()
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_loss = -float(losses.sum())
    return {
        "trade_count": int(len(returns)),
        "trade_win_rate": float((returns > 0).mean()),
        "avg_trade_return": float(returns.mean()),
        "median_trade_return": float(returns.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "add_count": int(positions["added"].fillna(False).astype(bool).sum()) if "added" in positions.columns else 0,
        "add_rate": float(positions["added"].fillna(False).astype(bool).mean()) if "added" in positions.columns else 0.0,
        "avg_holding_days": float(positions["holding_days"].mean()),
        "profit_factor": float(wins.sum()) / gross_loss if gross_loss > 0 else 0.0,
    }


def count_blocked_average_down(skips: pd.DataFrame) -> int:
    if skips.empty or "reason" not in skips.columns:
        return 0
    reasons = skips["reason"].astype(str)
    return int(reasons.str.startswith("average_down_blocked").sum())


def build_report(config: StockDeep50Config, signals: pd.DataFrame, result: dict) -> str:
    eq_stats = equity_stats(result["equity"], config.initial_capital)
    pos_stats = position_stats(result["positions"])
    blocked_add_count = count_blocked_average_down(result["skips"])
    lines = [
        "deep50 个股增强独立回测",
        "=" * 100,
        f"区间: {config.start_date.date()} ~ {config.end_date.date()}",
        f"总资金: {config.initial_capital:,.2f}",
        f"个股增强仓: {config.stock_sleeve_ratio * 100:.0f}% = {config.stock_sleeve_cash:,.2f}",
        f"单票金额: {config.position_cash:,.2f} = 个股增强仓 / {config.max_positions}",
        f"持仓: max_positions={config.max_positions}, max_daily_buy={config.max_daily_buy}",
        f"退出: +{config.take_profit_pct * 100:.1f}% 盘中定价止盈 / -{config.stop_loss_pct * 100:.1f}% 收盘止损次日开盘 / {config.max_holding_days}日时间止损",
        f"补仓: 盘中浮亏 -{config.add_trigger_pct * 100:.1f}% 只生成补仓信号；收盘确认当日 down_rate < {config.max_market_down_rate_for_add * 100:.0f}% 后，次日开盘等额补仓一次",
        f"排序: rank_score 从高到低，symbol 升序兜底，非随机",
        "-" * 100,
        f"deep50 信号数: {len(signals)}",
        f"已完成交易: {pos_stats['trade_count']}",
        f"期末持仓: {len(result['open_positions'])}",
        f"总收益(按总资金): {pct(eq_stats.get('total_return', np.nan))}",
        f"年化收益(按总资金): {pct(eq_stats.get('annualized_return', np.nan))}",
        f"最大回撤(按总资金): {pct(eq_stats.get('max_drawdown', np.nan))}",
        f"42日滚动盈利率: {pct(eq_stats.get('win_42', np.nan))}",
        f"平均42日收益: {pct(eq_stats.get('avg_42', np.nan))}",
        f"平均资金占用: {pct(eq_stats.get('avg_capital_utilization', np.nan))}",
        f"最大持仓数: {eq_stats.get('max_position_count', 0)}",
        "-" * 100,
        f"单笔胜率: {pct(pos_stats['trade_win_rate'])}",
        f"单笔平均收益: {pct(pos_stats['avg_trade_return'])}",
        f"单笔中位收益: {pct(pos_stats['median_trade_return'])}",
        f"平均盈利: {pct(pos_stats.get('avg_win', np.nan))}",
        f"平均亏损: {pct(pos_stats.get('avg_loss', np.nan))}",
        f"Profit Factor: {pos_stats['profit_factor']:.2f}",
        f"平均持仓天数: {pos_stats['avg_holding_days']:.2f}",
        f"补仓交易数: {pos_stats['add_count']} | 补仓率: {pct(pos_stats['add_rate'])} | 市场情绪拦截补仓信号次数: {blocked_add_count}",
    ]
    return "\n".join(lines)


def sell_reason_label(reason: str) -> str:
    labels = {
        "tp_buy_day_next_open": "买入日盘中触达止盈，受 T+1 约束次日开盘卖出",
        "tp_add_day_next_open": "补仓日盘中触达止盈，整笔交易受 T+1 约束次日开盘卖出",
        "stop_loss": "收盘跌破止损线，次日开盘卖出",
        "time_stop": "持仓达到时间上限，次日开盘卖出",
    }
    return labels.get(reason, reason)


def latest_row_before_or_on(data: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    window = data[data["date"] <= date].copy()
    if window.empty:
        return None
    return window.iloc[-1]


def find_next_trade_date(market_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> Optional[pd.Timestamp]:
    future_dates = set()
    for data in market_data.values():
        if data.empty:
            continue
        future_dates.update(item for item in data.index.tolist() if item > date)
    if not future_dates:
        return None
    return min(future_dates)


def collect_live_deep50_candidates(
    market_data: Dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    peak_window: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for symbol, indexed_data in market_data.items():
        raw_data = indexed_data.reset_index(drop=True)
        raw_data = raw_data[raw_data["date"] <= signal_date].copy()
        if len(raw_data) < peak_window + 60:
            continue

        data = add_indicators(raw_data, peak_window)
        signal_row = latest_row_before_or_on(data, signal_date)
        if signal_row is None:
            continue

        row_date = pd.Timestamp(signal_row["date"]).normalize()
        if row_date != signal_date:
            continue

        drawdown_from_peak = float(signal_row["drawdown_from_peak"]) if pd.notna(signal_row["drawdown_from_peak"]) else np.nan
        if pd.isna(drawdown_from_peak) or drawdown_from_peak > -0.50:
            continue

        long_lower_shadow = bool(
            float(signal_row["lower_shadow_pct"]) >= 0.04
            and float(signal_row["intraday_rebound_pct"]) >= 0.04
        )
        big_bull_body = bool(float(signal_row["body_pct"]) >= 0.04)
        if not long_lower_shadow and not big_bull_body:
            continue

        pre20_return = float(signal_row["pre20_return"]) if pd.notna(signal_row["pre20_return"]) else 0.0
        pre20_down_ratio = float(signal_row["pre20_down_ratio"]) if pd.notna(signal_row["pre20_down_ratio"]) else 0.0
        rank_score = (
            min(max(-pre20_return, 0.0), 0.5) * 4.0
            + pre20_down_ratio * 2.0
            + float(long_lower_shadow)
            + float(big_bull_body)
            + min(max(-drawdown_from_peak, 0.0), 0.8)
        )
        rows.append(
            {
                "symbol": symbol,
                "signal_date": row_date,
                "signal_close": float(signal_row["close"]),
                "signal_type": "both" if long_lower_shadow and big_bull_body else ("long_lower_shadow" if long_lower_shadow else "big_bull_body"),
                "drawdown_from_peak": drawdown_from_peak,
                "pre20_return": pre20_return,
                "pre20_down_ratio": pre20_down_ratio,
                "signal_body_pct": float(signal_row["body_pct"]),
                "signal_lower_shadow_pct": float(signal_row["lower_shadow_pct"]),
                "signal_volume_ratio_20": float(signal_row["volume_ratio_20"]) if pd.notna(signal_row["volume_ratio_20"]) else np.nan,
                "rank_score": rank_score,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rank_score", "symbol"], ascending=[False, True])


def build_average_down_watchlist(
    config: StockDeep50Config,
    market_data: Dict[str, pd.DataFrame],
    open_positions: pd.DataFrame,
    pending_sells: Dict[str, str],
    pending_adds: Dict[str, str],
    last_date: pd.Timestamp,
) -> List[dict]:
    if open_positions.empty:
        return []

    rows = []
    pending_sell_symbols = set(pending_sells.keys())
    pending_add_symbols = set(pending_adds.keys())
    for _, position in open_positions.iterrows():
        symbol = normalize_symbol(position["symbol"])
        if symbol in pending_sell_symbols or symbol in pending_add_symbols:
            continue

        added = bool(position.get("added", False))
        if added:
            continue

        buy_price = float(position.get("buy_price", 0.0))
        if buy_price <= EPS:
            continue

        row = latest_row_before_or_on(market_data.get(symbol, pd.DataFrame()).reset_index(drop=True), last_date)
        if row is None:
            continue

        last_close = float(row["close"])
        trigger_price = buy_price * (1.0 - config.add_trigger_pct)
        distance_to_trigger = last_close / trigger_price - 1.0 if trigger_price > EPS else np.nan
        rows.append(
            {
                "symbol": symbol,
                "buy_price": buy_price,
                "last_close": last_close,
                "trigger_price": trigger_price,
                "distance_to_trigger": distance_to_trigger,
                "add_cash": float(position.get("add_cash", config.position_cash * config.add_cash_multiplier)),
            }
        )

    return sorted(rows, key=lambda item: (item["distance_to_trigger"], item["symbol"]))


def build_pending_average_down_plan(
    config: StockDeep50Config,
    open_positions: pd.DataFrame,
    pending_adds: Dict[str, str],
    pending_sells: Dict[str, str],
) -> List[dict]:
    if open_positions.empty or not pending_adds:
        return []

    rows = []
    pending_sell_symbols = set(pending_sells.keys())
    for _, position in open_positions.iterrows():
        symbol = normalize_symbol(position["symbol"])
        if symbol not in pending_adds or symbol in pending_sell_symbols:
            continue

        buy_price = float(position.get("buy_price", 0.0))
        if buy_price <= EPS:
            continue

        rows.append(
            {
                "symbol": symbol,
                "trigger_price": buy_price * (1.0 - config.add_trigger_pct),
                "add_cash": float(position.get("add_cash", config.position_cash * config.add_cash_multiplier)),
                "reason": pending_adds[symbol],
            }
        )

    return sorted(rows, key=lambda item: item["symbol"])


def build_tomorrow_plan(
    config: StockDeep50Config,
    market_data: Dict[str, pd.DataFrame],
    result: dict,
    peak_window: int,
) -> str:
    last_date = pd.Timestamp(result.get("last_date")).normalize()
    pending_sells: Dict[str, str] = result.get("pending_sells", {})
    pending_adds: Dict[str, str] = result.get("pending_adds", {})
    open_positions = result.get("open_positions", pd.DataFrame())
    held_symbols = set(open_positions["symbol"].astype(str)) if not open_positions.empty else set()
    sell_symbols = set(pending_sells.keys())
    next_trade_date = find_next_trade_date(market_data, last_date)
    candidates = collect_live_deep50_candidates(market_data, last_date, peak_window)
    add_plan = build_pending_average_down_plan(config, open_positions, pending_adds, pending_sells)
    add_watchlist = build_average_down_watchlist(config, market_data, open_positions, pending_sells, pending_adds, last_date)

    available_slots = max(config.max_positions - len(held_symbols) + len(sell_symbols), 0)
    buy_limit = min(config.max_daily_buy, available_slots)
    if not candidates.empty:
        candidates = candidates[~candidates["symbol"].isin(held_symbols)].head(buy_limit).copy()

    next_date_text = next_trade_date.date().isoformat() if next_trade_date is not None else "缓存中暂无下一交易日"
    lines = [
        "",
        "=" * 100,
        f"明日作战计划（基于 {last_date.date().isoformat()} 收盘数据）",
        "=" * 100,
        f"计划执行日: {next_date_text}",
        f"单票计划金额: {config.position_cash:,.2f}",
        "",
        "卖出计划",
        "-" * 100,
    ]

    if not pending_sells:
        lines.append("无")
    else:
        for symbol, reason in sorted(pending_sells.items()):
            lines.append(f"{symbol} | {sell_reason_label(reason)} | 若开盘跌停无法卖出，则顺延到后续交易日")

    lines.extend(["", "补仓计划", "-" * 100])
    if not add_plan:
        lines.append("无")
    else:
        for row in add_plan:
            lines.append(
                f"{row['symbol']} | 补仓金额 {row['add_cash']:,.2f} | "
                f"触发价<={row['trigger_price']:.3f} | "
                f"{last_date.date().isoformat()} 已触发并通过 down_rate 约束 | "
                f"{next_date_text} 开盘执行"
            )

    lines.extend(["", "补仓观察", "-" * 100])
    if not add_watchlist:
        lines.append("无")
    else:
        for row in add_watchlist:
            lines.append(
                f"{row['symbol']} | 补仓金额 {row['add_cash']:,.2f} | "
                f"触发价<={row['trigger_price']:.3f} | "
                f"最新收盘 {row['last_close']:.3f} | "
                f"距触发 {pct(row['distance_to_trigger'])} | "
                f"执行规则: 盘中触发后，收盘确认 down_rate < {config.max_market_down_rate_for_add * 100:.0f}%，下一交易日开盘补仓"
            )

    lines.extend(["", "买入候选", "-" * 100])
    if candidates.empty:
        lines.append("无")
    else:
        for _, row in candidates.iterrows():
            lines.append(
                f"{row['symbol']} | 金额 {config.position_cash:,.2f} | "
                f"rank={row['rank_score']:.3f} | "
                f"回撤={pct(row['drawdown_from_peak'])} | "
                f"20日跌幅={pct(row['pre20_return'])} | "
                f"20日阴线密度={pct(row['pre20_down_ratio'])} | "
                f"信号={row['signal_type']}"
            )

    lines.extend(
        [
            "",
            "说明",
            "-" * 100,
            "买入候选只使用截至信号日收盘的日线数据，不依赖未来行情。",
            "补仓计划是已经触发并完成收盘确认的次日开盘操作；补仓观察只是条件单提示，不会在触发当天成交。",
            "同日多票按 rank_score 从高到低排序，再按 symbol 升序兜底，不随机。",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-csv", default=str(DEFAULT_SIGNAL_CSV))
    parser.add_argument("--market-cache", default=str(DEFAULT_MARKET_CACHE))
    parser.add_argument("--stock-list-csv", default=str(DEFAULT_STOCK_LIST_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--stock-sleeve-ratio", type=float, default=0.30)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-daily-buy", type=int, default=3)
    parser.add_argument("--take-profit-pct", type=float, default=0.08)
    parser.add_argument("--stop-loss-pct", type=float, default=0.10)
    parser.add_argument("--max-holding-days", type=int, default=10)
    parser.add_argument("--add-trigger-pct", type=float, default=0.10)
    parser.add_argument("--add-cash-multiplier", type=float, default=1.0)
    parser.add_argument("--max-market-down-rate-for-add", type=float, default=0.65)
    parser.add_argument("--tdx-dir", default=os.getenv("TDX_DIR", "/app/tdx_data"))
    parser.add_argument("--cache-start", default="2015-01-01")
    parser.add_argument("--cache-end", default="", help="为空时使用 --end；日期推进后传新的 --end 即可自动增量合并")
    parser.add_argument("--cache-max-symbols", type=int, default=0, help="0 为全量；仅用于缓存构建 smoke test")
    parser.add_argument("--cache-min-rows", type=int, default=120)
    parser.add_argument("--cache-adjust", choices=["raw", "hfq_local", "hfq"], default="hfq_local")
    parser.add_argument("--factor-cache-dir", default=str(DEFAULT_FACTOR_CACHE_DIR))
    parser.add_argument("--allow-online-adjust", action="store_true")
    parser.add_argument("--force-refresh-cache", action="store_true")
    parser.add_argument("--refresh-signals", action="store_true")
    parser.add_argument("--no-auto-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--cache-info", action="store_true")
    parser.add_argument("--peak-window", type=int, default=120)
    parser.add_argument("--signal-gap-days", type=int, default=20)
    parser.add_argument("--no-tomorrow-plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    cache_start = pd.Timestamp(args.cache_start)
    cache_end = pd.Timestamp(args.cache_end) if args.cache_end else end_date
    market_cache_path = Path(args.market_cache)

    if args.cache_info:
        print_market_cache_info(market_cache_path)
        return

    if not args.no_auto_cache:
        market_cache_path = ensure_mootdx_market_cache(
            market_cache_path,
            Path(args.stock_list_csv),
            args.tdx_dir,
            cache_start,
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
            return

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

    config = StockDeep50Config(
        initial_capital=args.initial_capital,
        stock_sleeve_ratio=args.stock_sleeve_ratio,
        max_positions=args.max_positions,
        max_daily_buy=args.max_daily_buy,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_holding_days=args.max_holding_days,
        add_trigger_pct=args.add_trigger_pct,
        add_cash_multiplier=args.add_cash_multiplier,
        max_market_down_rate_for_add=args.max_market_down_rate_for_add,
        start_date=start_date,
        end_date=end_date,
    )
    engine = StockDeep50Backtest(config, signals, market_data)
    result = engine.run()

    signals.to_csv(output_dir / "deep50_signals.csv", index=False, encoding="utf-8-sig")
    result["equity"].to_csv(output_dir / "equity.csv", index=False, encoding="utf-8-sig")
    result["positions"].to_csv(output_dir / "positions.csv", index=False, encoding="utf-8-sig")
    result["trades"].to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    result["skips"].to_csv(output_dir / "skips.csv", index=False, encoding="utf-8-sig")
    result["open_positions"].to_csv(output_dir / "open_positions.csv", index=False, encoding="utf-8-sig")

    report = build_report(config, signals, result)
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    if not args.no_tomorrow_plan:
        tomorrow_plan = build_tomorrow_plan(config, market_data, result, args.peak_window)
        (output_dir / "tomorrow_plan.txt").write_text(tomorrow_plan, encoding="utf-8")
        print(tomorrow_plan)
    print(f"\n已导出: {output_dir}")


if __name__ == "__main__":
    main()
