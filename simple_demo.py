#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通达信离线数据读取 - 简单示例
============================

这是一个简化的示例，演示mootdx库的基本用法
"""

from mootdx.reader import Reader
import pandas as pd

def main():
    """简单示例主函数"""
    
    print("通达信离线数据读取 - 简单示例")
    print("=" * 40)
    
    # 1. 初始化Reader
    # 请根据实际情况修改通达信安装目录
    tdx_dir = 'D:/通达信'  # 使用测试数据
    # tdx_dir = 'C:/new_tdx'    # 实际通达信安装目录
    
    try:
        # 创建标准市场读取器
        reader = Reader.factory(market='std', tdxdir=tdx_dir)
        print(f"✓ 成功连接到数据目录: {tdx_dir}")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return
    
    # 2. 读取日线数据
    print("\n📈 读取日线数据示例:")
    symbol = '000001'  # 平安银行
    
    try:
        daily_data = reader.daily(symbol=symbol)
        
        if daily_data is not None and not daily_data.empty:
            print(f"✓ 成功读取 {symbol} 日线数据")
            print(f"  数据条数: {len(daily_data)}")
            print(f"  数据列: {list(daily_data.columns)}")
            print(f"  最新5条数据:")
            print(daily_data.tail())
        else:
            print(f"✗ 未找到 {symbol} 的数据")
            
    except Exception as e:
        print(f"✗ 读取失败: {e}")
    
    # 3. 读取前复权数据
    print(f"\n📊 读取 {symbol} 前复权数据:")
    try:
        qfq_data = reader.daily(symbol=symbol, adjust='qfq')
        
        if qfq_data is not None and not qfq_data.empty:
            print(f"✓ 成功读取前复权数据，共 {len(qfq_data)} 条")
            print("  最新5条数据:")
            print(qfq_data.tail())
        else:
            print("✗ 未找到前复权数据")
            
    except Exception as e:
        print(f"✗ 读取前复权数据失败: {e}")
    
    # 4. 读取分钟数据
    # print(f"\n⏰ 读取 {symbol} 1分钟数据:")
    # try:
    #     minute_data = reader.minute(symbol=symbol, suffix=1)
        
    #     if minute_data is not None and not minute_data.empty:
    #         print(f"✓ 成功读取1分钟数据，共 {len(minute_data)} 条")
    #         print("  最新5条数据:")
    #         print(minute_data.tail())
    #     else:
    #         print("✗ 未找到1分钟数据")
            
    # except Exception as e:
        print(f"✗ 读取1分钟数据失败: {e}")
    
    # 5. 读取5分钟数据
    # print(f"\n⏰ 读取 {symbol} 5分钟数据:")
    # try:
    #     fzline_data = reader.fzline(symbol=symbol)
        
    #     if fzline_data is not None and not fzline_data.empty:
    #         print(f"✓ 成功读取5分钟数据，共 {len(fzline_data)} 条")
    #         print("  最新5条数据:")
    #         print(fzline_data.tail())
    #     else:
    #         print("✗ 未找到5分钟数据")
            
    # except Exception as e:
        print(f"✗ 读取5分钟数据失败: {e}")
    
    # 6. 数据导出示例
    print(f"\n💾 数据导出示例:")
    try:
        if 'daily_data' in locals() and not daily_data.empty:
            # 导出为CSV文件
            filename = f'{symbol}_daily_data.csv'
            daily_data.to_csv(filename, encoding='utf-8-sig')
            print(f"✓ 数据已导出到: {filename}")
        else:
            print("✗ 没有可导出的数据")
            
    except Exception as e:
        print(f"✗ 导出失败: {e}")
    
    print("\n" + "=" * 40)
    print("示例运行完成!")
    print("\n使用说明:")
    print("1. 请确保通达信客户端已安装并下载了数据")
    print("2. 修改 tdx_dir 变量为实际的通达信安装目录")
    print("3. 常见目录: C:/new_tdx, D:/new_tdx, C:/tdx")
    print("4. 更多功能请查看完整demo: demo_offline_reader.py")


if __name__ == '__main__':
    main()
