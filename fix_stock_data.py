import pandas as pd
from pathlib import Path
from mootdx.reader import Reader

tdx = "/app/tdx_data"
cache_dir = Path("/app/workspace/data_cache")
reader = Reader.factory(market="std", tdxdir=tdx)

for sym in ["513500", "511880", "510880","518880"]:
    cache_file = cache_dir / f"{sym}_hfq_cache.csv"
    c = pd.read_csv(cache_file)
    c["date"] = pd.to_datetime(c["date"])

    raw = reader.daily(symbol=sym).reset_index()
    dcol = next(col for col in ["date","datetime","Date"] if col in raw.columns)
    raw[dcol] = pd.to_datetime(raw[dcol])
    raw = raw.rename(columns={dcol:"date"})
    for col in ["open","high","low","close"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["date","open","high","low","close"]).sort_values("date")

    last_cache_date = c["date"].max()
    cache_last = c[c["date"] == last_cache_date].iloc[-1]
    raw_last = raw[raw["date"] == last_cache_date].iloc[-1]
    ratio = float(cache_last["close"]) / float(raw_last["close"])

    need = raw[raw["date"] > last_cache_date].copy()
    if need.empty:
        print(sym, "无需补数")
        continue

    for col in ["open","high","low","close"]:
        need[col] = need[col] * ratio

    out = pd.concat([c, need[c.columns.intersection(need.columns).tolist()]], ignore_index=True)
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    out.to_csv(cache_file, index=False)
    print(sym, "已补到", out["date"].max().date(), "ratio=", ratio)