#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机构级全天候 ETF-only - 滚动窗口极速压测工具
"""

import pandas as pd
from all_weather_strategy import AllWeatherEngine

# ================= 配置区 =================
WINDOW_YEARS = 1       # 窗口期：1年
STEP_DAYS = 2          # 步长：每次滑动 2 个交易日
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

def run_rolling_test():
    print("🚀 启动【全天候 ETF-only 三 ETF 引擎】滚动窗口压测...")
    print(f"⚙️ 参数: 窗口期={WINDOW_YEARS}年, 滑动步长={STEP_DAYS}交易日\n")
    
    # 1. 启动“母体引擎”，一次性预加载 10 年的所有底层数据
    master_engine = AllWeatherEngine(START_DATE, END_DATE)
    master_engine._prepare_data()
    
    calendar = master_engine.trade_calendar
    total_days = len(calendar)
    window_bars = int(WINDOW_YEARS * 252)
    
    if total_days <= window_bars:
        print("❌ 数据周期不足以进行滚动回测。")
        return

    # 2. 生成滚动切片
    start_indices = range(0, total_days - window_bars, STEP_DAYS)
    total_runs = len(start_indices)
    print(f"\n🔬 数据注入完毕！计划执行 {total_runs} 次 1年期 滚动平行宇宙测试...\n")
    
    results = []
    
    # 3. 开始极速滚动回测
    for idx, i in enumerate(start_indices):
        s_dt = calendar[i]
        e_dt = calendar[i + window_bars - 1]
        s_str = s_dt.strftime("%Y-%m-%d")
        e_str = e_dt.strftime("%Y-%m-%d")
        
        print(f"[{idx+1}/{total_runs}] 正在穿越平行宇宙: {s_str} -> {e_str} ...", end="\r")
        
        # 创建子窗口引擎
        win_engine = AllWeatherEngine(s_str, e_str)
        
        # 🔑 黑科技：内存指针注入 (绕过磁盘和网络，直接共享母体数据)
        win_engine.etf_data = master_engine.etf_data
        win_engine.stock_data = {}
        win_engine.stock_candidates_by_date = {}
        win_engine.symbol_name_map = getattr(master_engine, "symbol_name_map", {})
        win_engine.trade_calendar = [d for d in calendar if s_dt <= d <= e_dt]
        
        # 霸占预加载方法，使其变为空函数 (支持吞噬任意参数)
        win_engine._prepare_data = lambda *args, **kwargs: None
        
        # 闭嘴运行 (verbose=False)
        res = win_engine.run(verbose=False)
        
        if res:
            results.append({
                "入场时间": s_str,
                "离场时间": e_str,
                "年化收益(%)": res["annualized_return"] * 100,
                "最大回撤(%)": res["max_drawdown"] * 100
            })

    print("\n\n🎉 滚动压测完成！统计分析结果如下：\n")
    
    df_res = pd.DataFrame(results)
    
    # 4. 统计终极报告
    win_rate = (df_res["年化收益(%)"] > 0).mean() * 100
    mean_return = df_res["年化收益(%)"].mean()
    min_return = df_res["年化收益(%)"].min()
    max_return = df_res["年化收益(%)"].max()
    
    mean_dd = df_res["最大回撤(%)"].mean()
    max_dd = df_res["最大回撤(%)"].min()
    
    print("=====================================================================")
    print(f"📊 【全天候ETF-only | {WINDOW_YEARS}年期】滚动回测 (共 {total_runs} 个样本)")
    print("=====================================================================")
    print(f"🏆 绝对胜率 (盈利退出): {win_rate:.2f}%")
    print(f"📈 平均年化收益:        {mean_return:.2f}%")
    print(f"🟢 最佳进场年化收益:    {max_return:.2f}%")
    print(f"🔴 最差进场年化收益:    {min_return:.2f}%")
    print("-" * 69)
    print(f"🌊 平均最大回撤:        {mean_dd:.2f}%")
    print(f"💥 极端最大回撤:        {max_dd:.2f}%")
    print("=====================================================================\n")
    
    print("⚠️ 让你最难受的 20 个进场时机 (最差收益)：")
    worst = df_res.sort_values("年化收益(%)").head(500)
    for _, row in worst.iterrows():
        print(f"   进场: {row['入场时间']} -> 离场: {row['离场时间']} | 年化: {row['年化收益(%)']:.2f}% | 回撤: {row['最大回撤(%)']:.2f}%")

if __name__ == "__main__":
    run_rolling_test()
