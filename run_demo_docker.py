#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker环境下运行通达信离线数据读取Demo
===================================

本脚本专门为Docker环境设计，使用项目内置的测试数据
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("  ⚠️  警告: Python版本过低，建议使用3.8+")
    else:
        print("  ✓ Python版本符合要求")
    
    # 检查mootdx是否已安装
    try:
        import mootdx
        print(f"  ✓ mootdx已安装，版本: {mootdx.__version__}")
    except ImportError:
        print("  ✗ mootdx未安装")
        print("  请运行: pip install 'mootdx[all]'")
        return False
    
    # 检查测试数据目录
    test_data_dir = Path('tests/fixtures')
    if test_data_dir.exists():
        print(f"  ✓ 找到测试数据目录: {test_data_dir}")
    else:
        print(f"  ✗ 未找到测试数据目录: {test_data_dir}")
        return False
    
    return True

def run_simple_demo():
    """运行简单示例"""
    print("\n" + "="*50)
    print("🚀 运行简单示例")
    print("="*50)
    
    try:
        from mootdx.reader import Reader
        
        # 使用测试数据目录
        tdx_dir = 'tests/fixtures'
        reader = Reader.factory(market='std', tdxdir=tdx_dir)
        print(f"✓ 成功初始化Reader，数据目录: {tdx_dir}")
        
        # 测试读取数据
        test_symbols = ['000001', '127021', '688001']
        
        for symbol in test_symbols:
            print(f"\n📊 测试读取 {symbol} 数据...")
            
            # 读取日线数据
            daily_data = reader.daily(symbol=symbol)
            
            if daily_data is not None and not daily_data.empty:
                print(f"  ✓ 成功读取日线数据，共 {len(daily_data)} 条")
                print(f"  📈 数据时间范围: {daily_data.index[0]} 至 {daily_data.index[-1]}")
                print(f"  📋 数据列: {list(daily_data.columns)}")
                
                # 显示最新几条数据
                print("  📄 最新3条数据:")
                print(daily_data.tail(3).to_string())
                
                # 尝试导出数据
                try:
                    output_file = f'{symbol}_daily_docker.csv'
                    daily_data.to_csv(output_file, encoding='utf-8-sig')
                    print(f"  💾 数据已导出到: {output_file}")
                except Exception as e:
                    print(f"  ⚠️  导出失败: {e}")
                
                break  # 找到一个有数据的股票就停止
            else:
                print(f"  ⚠️  未找到 {symbol} 的数据")
        
        # 测试分钟数据
        print(f"\n⏰ 测试分钟数据读取...")
        for symbol in test_symbols:
            minute_data = reader.minute(symbol=symbol, suffix=1)
            if minute_data is not None and not minute_data.empty:
                print(f"  ✓ 成功读取 {symbol} 1分钟数据，共 {len(minute_data)} 条")
                break
            else:
                print(f"  ⚠️  未找到 {symbol} 的分钟数据")
        
        # 测试板块数据
        print(f"\n📋 测试板块数据读取...")
        try:
            block_data = reader.block(symbol='block_zs')
            if block_data is not None and not block_data.empty:
                print(f"  ✓ 成功读取板块数据，共 {len(block_data)} 条")
                print("  📄 前5条板块数据:")
                print(block_data.head().to_string())
            else:
                print("  ⚠️  未找到板块数据")
        except Exception as e:
            print(f"  ⚠️  读取板块数据失败: {e}")
        
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        return False
    
    return True

def run_extended_demo():
    """运行扩展示例"""
    print("\n" + "="*50)
    print("🔬 运行扩展示例")
    print("="*50)
    
    try:
        from mootdx.reader import Reader
        import pandas as pd
        
        # 初始化读取器
        tdx_dir = 'tests/fixtures'
        std_reader = Reader.factory(market='std', tdxdir=tdx_dir)
        ext_reader = Reader.factory(market='ext', tdxdir=tdx_dir)
        
        print("✓ 成功初始化标准市场和扩展市场读取器")
        
        # 测试复权数据
        print("\n📊 测试复权数据...")
        test_symbol = '000001'
        
        # 普通数据
        normal_data = std_reader.daily(symbol=test_symbol)
        if normal_data is not None and not normal_data.empty:
            print(f"  ✓ 普通数据: {len(normal_data)} 条")
            
            # 前复权数据
            qfq_data = std_reader.daily(symbol=test_symbol, adjust='qfq')
            if qfq_data is not None and not qfq_data.empty:
                print(f"  ✓ 前复权数据: {len(qfq_data)} 条")
                
                # 比较最后一条数据
                print("  📈 数据对比 (最新一条):")
                print(f"    普通收盘价: {normal_data['close'].iloc[-1]:.2f}")
                print(f"    前复权收盘价: {qfq_data['close'].iloc[-1]:.2f}")
            else:
                print("  ⚠️  前复权数据读取失败")
        
        # 测试扩展市场
        print("\n🔍 测试扩展市场数据...")
        ext_symbols = ['4#CF7D0LAO']  # 测试数据中的扩展市场代码
        
        for symbol in ext_symbols:
            ext_data = ext_reader.daily(symbol=symbol)
            if ext_data is not None and not ext_data.empty:
                print(f"  ✓ 成功读取扩展市场 {symbol} 数据，共 {len(ext_data)} 条")
                break
            else:
                print(f"  ⚠️  未找到扩展市场 {symbol} 数据")
        
        # 数据统计分析示例
        print("\n📈 数据统计分析示例...")
        if 'normal_data' in locals() and not normal_data.empty:
            print("  📊 基本统计信息:")
            print(f"    数据条数: {len(normal_data)}")
            print(f"    最高价: {normal_data['high'].max():.2f}")
            print(f"    最低价: {normal_data['low'].min():.2f}")
            print(f"    平均收盘价: {normal_data['close'].mean():.2f}")
            print(f"    总成交量: {normal_data['volume'].sum():,.0f}")
            
            # 计算简单移动平均线
            if len(normal_data) >= 20:
                normal_data['ma5'] = normal_data['close'].rolling(5).mean()
                normal_data['ma20'] = normal_data['close'].rolling(20).mean()
                
                latest_data = normal_data.iloc[-1]
                print(f"    最新收盘价: {latest_data['close']:.2f}")
                print(f"    5日均线: {latest_data['ma5']:.2f}")
                print(f"    20日均线: {latest_data['ma20']:.2f}")
        
    except Exception as e:
        print(f"✗ 扩展示例运行失败: {e}")
        return False
    
    return True

def show_data_structure():
    """显示数据结构信息"""
    print("\n" + "="*50)
    print("📁 数据结构信息")
    print("="*50)
    
    # 显示测试数据目录结构
    test_dir = Path('tests/fixtures')
    if test_dir.exists():
        print(f"📂 测试数据目录: {test_dir}")
        
        vipdoc_dir = test_dir / 'vipdoc'
        if vipdoc_dir.exists():
            print(f"  📂 vipdoc目录: {vipdoc_dir}")
            
            for market_dir in vipdoc_dir.iterdir():
                if market_dir.is_dir():
                    print(f"    📂 {market_dir.name}市场:")
                    
                    for sub_dir in market_dir.iterdir():
                        if sub_dir.is_dir():
                            file_count = len(list(sub_dir.glob('*')))
                            print(f"      📁 {sub_dir.name}: {file_count} 个文件")
                            
                            # 显示几个示例文件
                            files = list(sub_dir.glob('*'))[:3]
                            for file in files:
                                print(f"        📄 {file.name}")
                            if len(list(sub_dir.glob('*'))) > 3:
                                print(f"        ... 等")

def main():
    """主函数"""
    print("🐳 Docker环境 - 通达信离线数据读取Demo")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请检查配置")
        return
    
    # 显示数据结构
    show_data_structure()
    
    # 运行简单示例
    if not run_simple_demo():
        print("\n❌ 简单示例运行失败")
        return
    
    # 运行扩展示例
    if not run_extended_demo():
        print("\n❌ 扩展示例运行失败")
        return
    
    print("\n" + "="*60)
    print("🎉 Demo运行完成！")
    print("="*60)
    
    print("\n📝 说明:")
    print("- 本Demo使用项目内置的测试数据")
    print("- 在实际使用中，请替换为真实的通达信数据目录")
    print("- 如需使用真实数据，请修改 tdx_dir 变量")
    print("- 更多功能请参考完整文档")
    
    print("\n🔗 相关文件:")
    print("- demo_offline_reader.py: 完整功能演示")
    print("- simple_demo.py: 简化版本")
    print("- README_DEMO.md: 详细说明文档")
    
    print("\n📊 生成的文件:")
    csv_files = list(Path('.').glob('*.csv'))
    if csv_files:
        for file in csv_files:
            print(f"- {file}")
    else:
        print("- 无导出文件")

if __name__ == '__main__':
    main()
