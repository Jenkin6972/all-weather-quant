#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机构级全天候宏观对冲矩阵 (All-Weather ETF-only)
三核引擎：红利低波 + 黄金ETF + 标普500
"""

import os
import json
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import pandas as pd
import numpy as np
import requests

EPS = 1e-12
LOT_SIZE = 100
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")

# 统一状态机
STATE_EMPTY = 0
STATE_INIT = 1
STATE_RESCUE1 = 2
STATE_RESCUE2 = 3
STATE_FULL = 4


def normalize_symbol(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def is_main_board(symbol: str) -> bool:
    return normalize_symbol(symbol).startswith(MAIN_BOARD_PREFIXES)


def shares_for_cash(price: float, cash_budget: float) -> float:
    if price <= EPS or cash_budget <= EPS:
        return 0.0
    return cash_budget / price

# ================= 数据基建 (极速缓存 + 东方财富兜底) =================
class DataFetcher:
    def __init__(self, tdx_dir: str, cache_dir: str):
        self.tdx_dir = tdx_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._reader = None
        # 网络调度参数：单线程 + 固定间隔 + 随机抖动
        self.net_interval_sec = 0.25
        self.net_jitter_sec = 0.25
        # 指数退避参数：1s, 2s, 4s
        self.max_retry = 3
        self.backoff_base_sec = 1.0
        self.http_timeout_sec = 8

    def _init_reader(self):
        if self._reader is None:
            from mootdx.reader import Reader
            self._reader = Reader.factory(market="std", tdxdir=self.tdx_dir)

    def _throttle_network_request(self):
        sleep_sec = self.net_interval_sec + random.uniform(0.0, self.net_jitter_sec)
        time.sleep(max(0.0, sleep_sec))

    def _sleep_backoff(self, attempt_idx: int):
        wait_sec = self.backoff_base_sec * (2 ** attempt_idx)
        time.sleep(wait_sec)

    def load_hs300_symbols(self) -> List[str]:
        cache_path = self.cache_dir / "hs300_symbols_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass

        self._init_reader()
        block_df = self._reader.block(symbol="block_zs", group=False)
        hs300 = block_df[block_df["blockname"] == "沪深300"]
        if hs300.empty:
            hs300 = block_df[block_df["blockname"].str.contains("沪深300", na=False)]
        
        symbols = sorted(hs300["code"].astype(str).str.zfill(6).dropna().unique().tolist())
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(symbols, f, ensure_ascii=False)
        return symbols

    def load_symbol_name_map(self) -> Dict[str, str]:
        cache_path = self.cache_dir / "symbol_name_map_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass

        try:
            from mootdx import consts
            from mootdx.quotes import Quotes
            client = Quotes.factory(market="std", multithread=False, heartbeat=False, bestip=True, timeout=10)
            sh = client.stocks(market=consts.MARKET_SH)
            sz = client.stocks(market=consts.MARKET_SZ)
            merged = pd.concat([sh, sz], ignore_index=True)
            merged["code"] = merged["code"].astype(str).str.zfill(6)
            name_map = dict(zip(merged["code"], merged["name"]))
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(name_map, f, ensure_ascii=False)
            return name_map
        except: return {}

    def load_hfq_daily(self, symbol: str) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{symbol}_hfq_cache.csv"
        df_cache = pd.DataFrame()
        cache_max_date: Optional[pd.Timestamp] = None
        local_max_date: Optional[pd.Timestamp] = None
        last_date_str = "19900101"

        # 1) 先读缓存
        if cache_file.exists():
            try:
                df_cache = pd.read_csv(cache_file)
                df_cache["date"] = pd.to_datetime(df_cache["date"])
                if not df_cache.empty:
                    cache_max_date = df_cache["date"].max()
            except Exception:
                pass

        self._init_reader()

        # 2) 零网络试探：只读本地 day 最新日期
        try:
            raw_local = self._reader.daily(symbol=symbol)
            if raw_local is not None and not raw_local.empty:
                raw_local = raw_local.reset_index()
                date_col = next((c for c in ["date", "datetime", "Date"] if c in raw_local.columns), None)
                if date_col:
                    raw_local[date_col] = pd.to_datetime(raw_local[date_col])
                    local_max_date = raw_local[date_col].max()
                    # 缓存与本地 day 同步：直接返回，完全不联网
                    if cache_max_date is not None and cache_max_date >= local_max_date:
                        return df_cache.sort_values("date")
        except Exception:
            pass

        # 本地试探失败但有缓存：保守返回缓存，避免网络雪崩
        if local_max_date is None and not df_cache.empty:
            return df_cache.sort_values("date")

        # 3) 仅更新“落后于本地 day 最新日期”的缓存
        need_refresh = (cache_max_date is None) or (local_max_date is None) or (cache_max_date < local_max_date)
        if not need_refresh:
            return df_cache.sort_values("date")

        if cache_max_date is not None:
            last_date = cache_max_date - pd.Timedelta(days=5)
            last_date_str = last_date.strftime("%Y%m%d")

        df_new = pd.DataFrame()

        # 4) mootdx hfq：指数退避重试
        for attempt in range(self.max_retry):
            try:
                self._throttle_network_request()
                raw_hfq = self._reader.daily(symbol=symbol, adjust="hfq")
                if raw_hfq is not None and not raw_hfq.empty:
                    data = raw_hfq.reset_index().rename(columns={"datetime":"date", "Date":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close"})
                    data["date"] = pd.to_datetime(data["date"])
                    df_new = data[["date", "open", "high", "low", "close"]].dropna()
                    if not df_new.empty:
                        break
            except Exception:
                if attempt < self.max_retry - 1:
                    self._sleep_backoff(attempt)

        # 5) 东财兜底：单线程节流 + 指数退避
        if df_new.empty:
            for market_id in (1, 0):
                url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                params = {
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56",
                    "ut": "7eea3edcaed734bea9cbfc24409ed989",
                    "klt": "101",
                    "fqt": "2",
                    "beg": last_date_str,
                    "end": "20500101",
                    "lmt": "100000",
                    "secid": f"{market_id}.{symbol}",
                }
                for attempt in range(self.max_retry):
                    try:
                        self._throttle_network_request()
                        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
                        resp = requests.get(url, params=params, headers=headers, timeout=self.http_timeout_sec)
                        json_data = resp.json()
                        data_block = json_data.get("data")
                        if not data_block:
                            # 当前 market_id 无有效数据，不再重试此 market_id
                            break
                        klines = data_block.get("klines")
                        if klines:
                            out = pd.DataFrame([item.split(",") for item in klines], columns=["date", "open", "close", "high", "low", "vol"])
                            out["date"] = pd.to_datetime(out["date"])
                            for c in ["open", "high", "low", "close"]:
                                out[c] = pd.to_numeric(out[c], errors="coerce")
                            df_new = out[["date", "open", "high", "low", "close"]].dropna()
                            if not df_new.empty:
                                break
                        else:
                            # 有 data 但无 klines，视为无需更新
                            break
                    except Exception as e:
                        print(f"👉 [Debug-东财网络异常] {symbol} m{market_id} attempt{attempt + 1}: {e}")
                        if attempt < self.max_retry - 1:
                            self._sleep_backoff(attempt)
                if not df_new.empty:
                    break

        # 6) 拼合与回退
        if not df_new.empty:
            if not df_cache.empty:
                df_merged = pd.concat([df_cache, df_new])
                df_merged = df_merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            else:
                df_merged = df_new.sort_values("date")
            df_merged.to_csv(cache_file, index=False)
            return df_merged

        if not df_cache.empty:
            return df_cache.sort_values("date")
        return pd.DataFrame()

# ================= 策略引擎配置 =================
@dataclass
class ETFConfig:
    name: str; code: str
    grid_cash: List[float]
    loss_trigger: List[float]
    tp_trigger: List[float]

@dataclass
class StockConfig:
    sleeve_cash: float = 0.0
    max_positions: int = 0
    max_daily_buy: int = 0
    initial_fraction: float = 0.5
    add_trigger: float = -0.08
    take_profit: float = 0.15
    stop_loss: float = -0.12
    max_holding_days: int = 60
    candidate_csv: str = "/app/tdx_data/rebound_guerrilla_20150101_20251231/candidates/t1_drawdown_rebound_candidates.csv"
    market_cache_paths: List[str] = field(default_factory=lambda: [
        "/app/workspace/data_cache/market_data_all_a_2015-01-01_2024-08-31_hfq_360_329db0915e98.pkl",
        "/app/workspace/data_cache/market_data_main_board_2020-01-01_2025-12-31_hfq_200_c9e24d15a649.pkl",
    ])

    @property
    def position_budget(self) -> float:
        if self.max_positions <= 0:
            return 0.0
        return self.sleeve_cash / self.max_positions

    @property
    def initial_cash(self) -> float:
        return self.position_budget * self.initial_fraction

    @property
    def add_cash(self) -> float:
        return self.position_budget * (1.0 - self.initial_fraction)

@dataclass
class AllWeatherConfig:
    initial_capital: float = 100000.0
    etf_a: ETFConfig = field(default_factory=lambda: ETFConfig(
        name="红利低波", code="510880", grid_cash=[5000, 10000, 10000, 10000],
        loss_trigger=[-0.04, -0.08, -0.12], tp_trigger=[0.04, 0.02, 0.01, 0.01]))
    etf_b: ETFConfig = field(default_factory=lambda: ETFConfig(
        name="黄金ETF", code="518880", grid_cash=[5000, 5000, 10000, 15000],
        loss_trigger=[-0.02, -0.04, -0.08], tp_trigger=[0.03, 0.03, 0.02, 0.02]))
    etf_c: ETFConfig = field(default_factory=lambda: ETFConfig(
        name="标普500", code="513500", grid_cash=[5000, 5000, 15000, 10000],
        loss_trigger=[-0.05, -0.10, -0.15], tp_trigger=[0.08, 0.05, 0.03, 0.02]))
    stock_d: StockConfig = field(default_factory=StockConfig)
    mmf_code: str = "511880"

def calc_rsi(close: pd.Series, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def load_rebound_candidates(config: StockConfig, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    if config.sleeve_cash <= EPS or config.max_positions <= 0 or config.max_daily_buy <= 0:
        return pd.DataFrame()

    candidate_path = Path(os.getenv("REBOUND_CANDIDATE_CSV", config.candidate_csv))
    if not candidate_path.exists():
        return pd.DataFrame()

    data = pd.read_csv(candidate_path, dtype={"symbol": str})
    data["symbol"] = data["symbol"].map(normalize_symbol)
    for column in ["t1_date", "t2_date", "buy_date"]:
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()

    numeric_columns = [
        "pre20_down_day_ratio",
        "t2_ma60_bias",
        "t2_volume_ratio_20",
        "buy_price",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["symbol", "buy_date", "pre20_down_day_ratio", "t2_ma60_bias"])
    data = data[
        (data["buy_date"] >= start_dt)
        & (data["buy_date"] <= end_dt)
        & (data["t2_ma60_bias"] < -0.25)
        & (data["pre20_down_day_ratio"] >= 0.60)
    ].copy()
    if data.empty:
        return data

    data["rank_score"] = data["pre20_down_day_ratio"]
    return data.sort_values(["buy_date", "rank_score", "symbol"], ascending=[True, False, True])


def normalize_daily_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    data.columns = [str(column).lower() for column in data.columns]
    if "date" not in data.columns:
        data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"])
    if data.empty:
        return data
    return data.sort_values("date").drop_duplicates("date", keep="last")


def load_stock_market_caches(paths: List[str], needed_symbols: Set[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    market_data: Dict[str, pd.DataFrame] = {}
    for raw_path in paths:
        cache_path = Path(os.getenv("STOCK_MARKET_CACHE", raw_path))
        if not cache_path.exists():
            continue
        with cache_path.open("rb") as file:
            raw_market_data = pickle.load(file)

        for raw_symbol, raw_data in raw_market_data.items():
            symbol = normalize_symbol(raw_symbol)
            if symbol not in needed_symbols or raw_data is None or raw_data.empty:
                continue
            data = normalize_daily_frame(raw_data)
            if data.empty:
                continue
            data = data[(data["date"] >= start_dt) & (data["date"] <= end_dt)].copy()
            if data.empty:
                continue
            if symbol in market_data:
                data = pd.concat([market_data[symbol].reset_index(drop=True), data], ignore_index=True)
                data = data.sort_values("date").drop_duplicates("date", keep="last")
            market_data[symbol] = data

    output = {}
    for symbol, data in market_data.items():
        data = data.sort_values("date").drop_duplicates("date", keep="last")
        data["prev_close"] = data["close"].shift(1)
        output[symbol] = data.set_index("date", drop=False)
    return output

class AllWeatherEngine:
    def __init__(self, start_date: str, end_date: str):
        self.config = AllWeatherConfig()
        self.fetcher = DataFetcher(tdx_dir=os.getenv("TDX_DIR", "/app/tdx_data"), cache_dir="/app/workspace/data_cache")
        self.start_dt = pd.to_datetime(start_date)
        self.end_dt = pd.to_datetime(end_date)
        
        self.etf_data = {}
        self.stock_data = {}
        self.stock_candidates_by_date = {}
        self.trade_calendar = []

        self.cash = self.config.initial_capital
        self.mmf_shares = 0.0
        self.mmf_last_price = 100.0  # 增加货基的记忆托底价格
        
        self.pos = {
            "A": {"shares": 0.0, "cost": 0.0, "state": STATE_EMPTY, "last_close": 0.0},
            "B": {"shares": 0.0, "cost": 0.0, "state": STATE_EMPTY, "last_close": 0.0},
            "C": {"shares": 0.0, "cost": 0.0, "state": STATE_EMPTY, "last_close": 0.0},
        }
        self.stock_positions = {}
        self.log_history = []

    def _log(self, date: str, msg: str):
        self.log_history.append(f"{date} {msg}")

    def _stock_sleeve_enabled(self) -> bool:
        stock_config = self.config.stock_d
        return (
            stock_config.sleeve_cash > EPS
            and stock_config.max_positions > 0
            and stock_config.max_daily_buy > 0
        )

    def _prepare_data(self, verbose: bool = True):
        if verbose: print("⏳ 正在预加载 ETF-only 全量数据...")
        cal_set = set()
        for key, conf in zip(["A", "B", "C"], [self.config.etf_a, self.config.etf_b, self.config.etf_c]):
            df = self.fetcher.load_hfq_daily(conf.code)
            if df.empty:
                if verbose: print(f"❌ 警告：无法获取 {conf.name}({conf.code}) 的数据，请检查网络。")
                continue
            df = df.set_index("date")
            df["ma20"] = df["close"].rolling(20).mean()
            df = df[(df.index >= self.start_dt) & (df.index <= self.end_dt)]
            self.etf_data[key] = df
            cal_set.update(df.index)
        
        self.etf_data["MMF"] = self.fetcher.load_hfq_daily(self.config.mmf_code).set_index("date")
        
        if self._stock_sleeve_enabled():
            self.symbol_name_map = self.fetcher.load_symbol_name_map()
            stock_candidates = load_rebound_candidates(self.config.stock_d, self.start_dt, self.end_dt)
            if stock_candidates.empty:
                if verbose: print("⚠️ 未加载到深跌反转游击队候选，游击队将保持空仓。")
            else:
                needed_symbols = set(stock_candidates["symbol"].dropna().astype(str).map(normalize_symbol))
                self.stock_data = load_stock_market_caches(
                    self.config.stock_d.market_cache_paths,
                    needed_symbols,
                    self.start_dt,
                    self.end_dt,
                )
                valid_symbols = set(self.stock_data.keys())
                stock_candidates = stock_candidates[stock_candidates["symbol"].isin(valid_symbols)].copy()
                self.stock_candidates_by_date = {
                    date: group.sort_values(["rank_score", "symbol"], ascending=[False, True]).copy()
                    for date, group in stock_candidates.groupby("buy_date")
                }
                for data in self.stock_data.values():
                    cal_set.update(data.index)
                if verbose:
                    print(f"✅ 深跌反转候选 {len(stock_candidates)} 条，涉及行情标的 {len(self.stock_data)} 只。")
                
        self.trade_calendar = sorted(list(cal_set))
        if verbose: print("✅ 数据加载完成！")

    def _ensure_cash(self, needed: float, date: pd.Timestamp) -> bool:
        if self.cash >= needed - EPS:
            return True
            
        # 实时更新并使用被记忆的货基价格
        if date in self.etf_data["MMF"].index:
            self.mmf_last_price = self.etf_data["MMF"].loc[date, "open"]
            
        shortage = needed - self.cash
        sell_shares = min(shortage / self.mmf_last_price, self.mmf_shares)
        self.cash += sell_shares * self.mmf_last_price
        self.mmf_shares -= sell_shares
        return self.cash >= needed - EPS

    def _sweep_idle_cash(self, date: pd.Timestamp):
        if self.cash > EPS:
            if date in self.etf_data["MMF"].index:
                self.mmf_last_price = self.etf_data["MMF"].loc[date, "open"]
            self.mmf_shares += self.cash / self.mmf_last_price
            self.cash = 0.0

    def _stock_holding_days(self, position: Dict[str, Any], date: pd.Timestamp) -> int:
        data = self.stock_data.get(position["symbol"])
        if data is None:
            return 0
        mask = (data.index >= position["entry_date"]) & (data.index <= date)
        return int(mask.sum())

    def _is_limit_down_open(self, row: pd.Series) -> bool:
        prev_close = float(row["prev_close"]) if pd.notna(row.get("prev_close")) else 0.0
        open_price = float(row["open"]) if pd.notna(row.get("open")) else 0.0
        if prev_close <= EPS or open_price <= EPS:
            return False
        return open_price / prev_close - 1.0 <= -0.095 + EPS

    def _execute_stock_actions(self, date: pd.Timestamp, actions: List[dict]) -> List[dict]:
        if not actions:
            return []

        date_str = date.strftime("%Y-%m-%d")
        remaining_actions = []
        buy_count = 0

        actions = sorted(actions, key=lambda item: {"sell": 0, "buy_add": 1, "buy_init": 2}.get(item["action"], 9))
        for action in actions:
            sym = action["symbol"]
            row = self.stock_data.get(sym, pd.DataFrame()).loc[date] if date in self.stock_data.get(sym, pd.DataFrame()).index else None
            name = self.symbol_name_map.get(sym, sym)

            if row is None:
                if action["action"] == "sell":
                    remaining_actions.append(action)
                continue

            open_price = float(row["open"])
            if action["action"] == "sell":
                position = self.stock_positions.get(sym)
                if not position:
                    continue
                if self._is_limit_down_open(row):
                    remaining_actions.append(action)
                    self._log(date_str, f"[游击队卖出顺延] {name} 开盘跌停，卖出计划顺延")
                    continue
                proceeds = position["shares"] * open_price
                pnl = open_price / position["cost"] - 1.0 if position["cost"] > EPS else 0.0
                self.cash += proceeds
                reason = action.get("reason", "卖出")
                self._log(date_str, f"[游击队{reason}] {name}，回笼 {proceeds:.0f}，收益 {pnl*100:.2f}%")
                del self.stock_positions[sym]
                continue

            if action["action"] == "buy_add":
                position = self.stock_positions.get(sym)
                if not position or position.get("add_done"):
                    continue
                add_cash = self.config.stock_d.add_cash
                if not self._ensure_cash(add_cash, date):
                    continue
                shares = shares_for_cash(open_price, add_cash)
                if shares <= 0:
                    continue
                cost = shares * open_price
                old_amount = position["shares"] * position["cost"]
                position["shares"] += shares
                position["cost"] = (old_amount + cost) / position["shares"]
                position["add_done"] = True
                position["last_close"] = open_price
                self.cash -= cost
                self._log(date_str, f"[游击队补仓] {name}，补入 {cost:.0f} 元，摊薄成本 {position['cost']:.3f}")
                continue

            if action["action"] == "buy_init":
                if buy_count >= self.config.stock_d.max_daily_buy:
                    continue
                if len(self.stock_positions) >= self.config.stock_d.max_positions:
                    continue
                if sym in self.stock_positions:
                    continue
                buy_cash = self.config.stock_d.initial_cash
                if not self._ensure_cash(buy_cash, date):
                    continue
                shares = shares_for_cash(open_price, buy_cash)
                if shares <= 0:
                    continue
                cost = shares * open_price
                self.cash -= cost
                self.stock_positions[sym] = {
                    "symbol": sym,
                    "name": name,
                    "shares": shares,
                    "cost": open_price,
                    "entry_date": date,
                    "last_close": open_price,
                    "add_done": False,
                }
                buy_count += 1
                score = action.get("pre20_down_day_ratio", 0.0)
                self._log(date_str, f"[游击队首发] {name}，买入 {cost:.0f} 元，20日阴线密度 {score*100:.0f}%")

        return remaining_actions

    def _generate_stock_actions(self, date: pd.Timestamp) -> List[dict]:
        actions: List[dict] = []

        for sym, position in list(self.stock_positions.items()):
            if date not in self.stock_data.get(sym, pd.DataFrame()).index:
                continue
            row = self.stock_data[sym].loc[date]
            close_price = float(row["close"])
            position["last_close"] = close_price
            pnl = close_price / position["cost"] - 1.0 if position["cost"] > EPS else 0.0
            holding_days = self._stock_holding_days(position, date)

            if pnl >= self.config.stock_d.take_profit:
                actions.append({"action": "sell", "symbol": sym, "pnl": pnl, "reason": "止盈"})
                continue
            if pnl <= self.config.stock_d.stop_loss:
                actions.append({"action": "sell", "symbol": sym, "pnl": pnl, "reason": "止损"})
                continue
            if holding_days >= self.config.stock_d.max_holding_days:
                actions.append({"action": "sell", "symbol": sym, "pnl": pnl, "reason": "时间止损"})
                continue
            if not position.get("add_done") and pnl <= self.config.stock_d.add_trigger:
                actions.append({"action": "buy_add", "symbol": sym, "pnl": pnl})

        if len(self.stock_positions) >= self.config.stock_d.max_positions:
            return actions

        candidates = self.stock_candidates_by_date.get(date)
        if candidates is None or candidates.empty:
            return actions

        buy_slots = self.config.stock_d.max_positions - len(self.stock_positions)
        buy_limit = min(buy_slots, self.config.stock_d.max_daily_buy)
        buy_added = 0
        for _, candidate in candidates.iterrows():
            if buy_added >= buy_limit:
                break
            sym = normalize_symbol(candidate["symbol"])
            if sym in self.stock_positions or sym not in self.stock_data:
                continue
            actions.append(
                {
                    "action": "buy_init",
                    "symbol": sym,
                    "pre20_down_day_ratio": float(candidate["pre20_down_day_ratio"]),
                }
            )
            buy_added += 1

        return actions

    def _dedupe_stock_actions(self, actions: List[dict]) -> List[dict]:
        output = []
        seen = set()
        for action in actions:
            key = (action.get("action"), action.get("symbol"), action.get("reason", ""))
            if key in seen:
                continue
            seen.add(key)
            output.append(action)
        return output

    def run(self, verbose: bool = True):
        self._prepare_data(verbose)
        stock_enabled = self._stock_sleeve_enabled()
        pending_etf = {"A": None, "B": None, "C": None}
        pending_stock = []
        
        equity_curve = []

        for date in self.trade_calendar:
            date_str = date.strftime("%Y-%m-%d")
            
            # ================= T+1 开盘执行 =================
            for key, conf in zip(["A", "B", "C"], [self.config.etf_a, self.config.etf_b, self.config.etf_c]):
                if pending_etf[key] is None or key not in self.etf_data or date not in self.etf_data[key].index: 
                    continue
                action = pending_etf[key]["action"]
                open_p = self.etf_data[key].loc[date, "open"]
                
                if action == "sell":
                    proceeds = self.pos[key]["shares"] * open_p
                    self.cash += proceeds
                    self._log(date_str, f"[{conf.name}止盈] 全仓清仓，回笼 {proceeds:.0f}，收益 {pending_etf[key]['pnl']*100:.2f}%")
                    self.pos[key] = {"shares": 0.0, "cost": 0.0, "state": STATE_EMPTY, "last_close": 0.0}
                
                elif action.startswith("buy_"):
                    level = int(action[-1]) if action[-1].isdigit() else 0
                    needed = conf.grid_cash[level]
                    if self._ensure_cash(needed, date):
                        shares = needed / open_p
                        old_val = self.pos[key]["shares"] * self.pos[key]["cost"]
                        self.pos[key]["shares"] += shares
                        self.pos[key]["cost"] = (old_val + needed) / self.pos[key]["shares"]
                        self.pos[key]["state"] += 1
                        self.pos[key]["last_close"] = open_p  # 买入时记录记忆价格
                        self.cash -= needed
                        self._log(date_str, f"[{conf.name}买入] 阶层{level}，消耗 {needed:.0f}，均价 {self.pos[key]['cost']:.3f}")

                pending_etf[key] = None

            if stock_enabled:
                pending_stock = self._execute_stock_actions(date, pending_stock)
            
            self._sweep_idle_cash(date)

            # ================= T 日收盘信号判定 =================
            for key, conf in zip(["A", "B", "C"], [self.config.etf_a, self.config.etf_b, self.config.etf_c]):
                if key not in self.etf_data or date not in self.etf_data[key].index: continue
                close_p = self.etf_data[key].loc[date, "close"]
                ma20 = self.etf_data[key].loc[date, "ma20"]
                state = self.pos[key]["state"]
                
                if state == STATE_EMPTY and close_p < ma20:
                    pending_etf[key] = {"action": "buy_0"}
                elif state > STATE_EMPTY:
                    pnl = close_p / self.pos[key]["cost"] - 1.0
                    tp_thresh = conf.tp_trigger[state-1]
                    if pnl >= tp_thresh:
                        pending_etf[key] = {"action": "sell", "pnl": pnl}
                    elif state < STATE_FULL:
                        loss_thresh = conf.loss_trigger[state-1]
                        if pnl <= loss_thresh:
                            pending_etf[key] = {"action": f"buy_{state}"}

            if stock_enabled:
                pending_stock.extend(self._generate_stock_actions(date))
                pending_stock = self._dedupe_stock_actions(pending_stock)

            # ================= 记录资产净值 (完美记忆防断层) =================
            total_val = self.cash
            
            if date in self.etf_data["MMF"].index:
                self.mmf_last_price = self.etf_data["MMF"].loc[date, "close"]
            total_val += self.mmf_shares * self.mmf_last_price
            
            for key in ["A", "B", "C"]:
                if date in self.etf_data[key].index:
                    self.pos[key]["last_close"] = self.etf_data[key].loc[date, "close"]
                if self.pos[key]["shares"] > 0:
                    val_price = self.pos[key]["last_close"] if self.pos[key]["last_close"] > 0 else self.pos[key]["cost"]
                    total_val += self.pos[key]["shares"] * val_price
                    
            if stock_enabled:
                for position in self.stock_positions.values():
                    if position["shares"] > 0:
                        total_val += position["shares"] * position["last_close"]
                
            equity_curve.append({"date": date, "equity": total_val})

        if not equity_curve:
            if verbose: print("❌ 回测失败，资金曲线为空！请检查数据加载。")
            return None
            
        df_eq = pd.DataFrame(equity_curve)
        df_eq["cummax"] = df_eq["equity"].cummax()
        df_eq["dd"] = df_eq["equity"] / df_eq["cummax"] - 1.0
        
        ar = (df_eq["equity"].iloc[-1] / self.config.initial_capital) ** (252 / len(df_eq)) - 1.0
        md = df_eq["dd"].min()
        
        if verbose:
            log_file = "/app/workspace/all_weather_trades.txt"
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.log_history))
            except Exception as e:
                print(f"日志文件写入失败: {e}")

            print("\n" + "="*80)
            print("🏆 全天候 ETF-only (三 ETF 网格引擎) 回测完成！")
            print("="*80)
            print(f"回测区间: {self.start_dt.date()} -> {self.end_dt.date()}")
            print(f"年化收益: {ar*100:.2f}%  |  最大回撤: {md*100:.2f}%")
            print(f"期末总值: {df_eq['equity'].iloc[-1]:,.2f} 元")
            print("-" * 80)
            print(f"📄 完整的交易流水已导出至: {log_file} (可打开该文件查看所有买卖、补仓和止损明细)")
            print("💡 近 10 条交易记录预览:")
            for msg in self.log_history[-10:]: print(msg)
            print("="*80 + "\n")
            
        return {
            "annualized_return": ar, 
            "max_drawdown": md,
            "last_date": df_eq.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "pending_etf": pending_etf,
            "pending_stock": pending_stock,
            "stock_positions": self.stock_positions,
        }

def print_tomorrow_plan(res: dict, config: AllWeatherConfig):
    if not res: return
    
    print("\n" + "=" * 80)
    print(f"🚀 明日作战计划 (基于 {res['last_date']} 收盘数据生成，将于下个交易日开盘执行)")
    print("=" * 80)
    
    has_action = False
    
    for key, conf in zip(["A", "B", "C"], [config.etf_a, config.etf_b, config.etf_c]):
        sig = res["pending_etf"].get(key)
        if sig:
            action = sig.get("action")
            if action == "sell":
                print(f"> 🔴 [{conf.name} 止盈] 已触发止盈条件 (收益率: {sig.get('pnl',0)*100:.2f}%)，计划开盘【全部清仓】！")
                has_action = True
            elif action.startswith("buy_"):
                level = int(action.split("_")[1])
                cash_needed = conf.grid_cash[level]
                if level == 0:
                    print(f"> 🟢 [{conf.name} 首发] 回撤达标，计划开盘卖出货基【买入 {cash_needed:,.0f} 元】！")
                else:
                    print(f"> 🟢 [{conf.name} 补仓] 浮亏深陷，计划打出阶层 {level} 救援弹药【买入 {cash_needed:,.0f} 元】！")
                has_action = True

    if config.stock_d.sleeve_cash > EPS and config.stock_d.max_positions > 0 and config.stock_d.max_daily_buy > 0:
        for sig_d in res.get("pending_stock", []):
            action = sig_d.get("action")
            sym = sig_d.get("symbol", "")
            if action == "sell":
                reason = sig_d.get("reason", "条件满足")
                print(f"> 🔴 [游击队卖出] {sym} 触发【{reason}】(收盘收益率: {sig_d.get('pnl',0)*100:.2f}%)，计划开盘【全部清仓】！")
                has_action = True
            elif action == "buy_init":
                down_rate = sig_d.get("pre20_down_day_ratio", 0.0)
                print(f"> 🟢 [游击队首发] {sym} 深跌反转入选，20日阴线密度 {down_rate*100:.0f}%，计划开盘【买入 {config.stock_d.initial_cash:,.0f} 元】！")
                has_action = True
            elif action == "buy_add":
                print(f"> 🟢 [游击队补仓] {sym} 浮亏达到 {config.stock_d.add_trigger*100:.0f}% 补仓线，计划开盘【补入 {config.stock_d.add_cash:,.0f} 元】！")
                has_action = True
            
    if not has_action:
        print("> ☕ 【平安无事】继续持有当前阵地 或 纯现金吃息 (暂无新操作信号)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    engine = AllWeatherEngine(start_date="2026-04-24", end_date="2026-10-08")
    res = engine.run(verbose=True)
    if res:
        print_tomorrow_plan(res, engine.config)
