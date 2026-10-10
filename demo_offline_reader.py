#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通达信离线数据读取Demo
===================

本示例演示如何使用mootdx库读取通达信客户端的离线数据，包括：
1. 日线数据读取
2. 分钟线数据读取  
3. 5分钟线数据读取
4. 板块数据读取
5. 数据导出功能
6. 复权处理

作者: mootdx项目
日期: 2025-01-16
"""

import os
import sys
from pathlib import Path
import pandas as pd
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

try:
    from mootdx.reader import Reader
    from mootdx.utils import to_file
    print("✓ mootdx库导入成功")
except ImportError as e:
    print(f"✗ 导入mootdx失败: {e}")
    print("请先安装mootdx: pip install 'mootdx[all]'")
    sys.exit(1)


class TdxOfflineReader:
    """通达信离线数据读取器"""
    
    def __init__(self, tdx_dir=None):
        """
        初始化读取器
        
        Args:
            tdx_dir (str): 通达信安装目录，如 'C:/new_tdx'
        """
        # 默认通达信安装路径
        default_paths = [
            'C:/new_tdx',
            'C:/tdx', 
            'D:/new_tdx',
            'D:/tdx',
            'tests/fixtures'  # 测试数据路径
        ]
        
        if tdx_dir:
            self.tdx_dir = tdx_dir
        else:
            # 自动查找通达信安装目录
            self.tdx_dir = self._find_tdx_dir(default_paths)
            
        if not self.tdx_dir:
            raise Exception("未找到通达信安装目录，请手动指定tdx_dir参数")
            
        print(f"✓ 使用通达信数据目录: {self.tdx_dir}")
        
        # 初始化标准市场和扩展市场读取器
        try:
            self.std_reader = Reader.factory(market='std', tdxdir=self.tdx_dir)
            self.ext_reader = Reader.factory(market='ext', tdxdir=self.tdx_dir)
            print("✓ 数据读取器初始化成功")
        except Exception as e:
            raise Exception(f"初始化读取器失败: {e}")
    
    def _find_tdx_dir(self, paths):
        """查找通达信安装目录"""
        for path in paths:
            if Path(path).exists():
                vipdoc_path = Path(path) / 'vipdoc'
                if vipdoc_path.exists():
                    return path
        return None
    
    def read_daily_data(self, symbol, adjust=None):
        """
        读取日线数据
        
        Args:
            symbol (str): 股票代码，如 '600036', '000001'
            adjust (str): 复权类型 'qfq'前复权, 'hfq'后复权, None不复权
            
        Returns:
            pd.DataFrame: 日线数据
        """
        try:
            print(f"\n📈 读取 {symbol} 日线数据...")
            
            # 读取日线数据
            data = self.std_reader.daily(symbol=symbol, adjust=adjust)
            
            if data is not None and not data.empty:
                print(f"✓ 成功读取 {len(data)} 条日线数据")
                print(f"  数据时间范围: {data.index[0]} 至 {data.index[-1]}")
                print(f"  数据列: {list(data.columns)}")
                return data
            else:
                print(f"✗ 未找到 {symbol} 的日线数据")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ 读取日线数据失败: {e}")
            return pd.DataFrame()
    
    def read_minute_data(self, symbol, suffix=1):
        """
        读取分钟线数据
        
        Args:
            symbol (str): 股票代码
            suffix (int): 1=1分钟线, 5=5分钟线
            
        Returns:
            pd.DataFrame: 分钟线数据
        """
        try:
            minute_type = "1分钟" if suffix == 1 else "5分钟"
            print(f"\n📊 读取 {symbol} {minute_type}线数据...")
            
            data = self.std_reader.minute(symbol=symbol, suffix=suffix)
            
            if data is not None and not data.empty:
                print(f"✓ 成功读取 {len(data)} 条{minute_type}线数据")
                print(f"  数据时间范围: {data.index[0]} 至 {data.index[-1]}")
                return data
            else:
                print(f"✗ 未找到 {symbol} 的{minute_type}线数据")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ 读取{minute_type}线数据失败: {e}")
            return pd.DataFrame()
    
    def read_fzline_data(self, symbol):
        """
        读取5分钟线数据（fzline方法）
        
        Args:
            symbol (str): 股票代码
            
        Returns:
            pd.DataFrame: 5分钟线数据
        """
        try:
            print(f"\n📊 读取 {symbol} 5分钟线数据(fzline)...")
            
            data = self.std_reader.fzline(symbol=symbol)
            
            if data is not None and not data.empty:
                print(f"✓ 成功读取 {len(data)} 条5分钟线数据")
                return data
            else:
                print(f"✗ 未找到 {symbol} 的5分钟线数据")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ 读取5分钟线数据失败: {e}")
            return pd.DataFrame()
    
    def read_ext_daily_data(self, symbol):
        """
        读取扩展市场日线数据（期货、期权等）
        
        Args:
            symbol (str): 扩展市场代码，如 '29#A1801'
            
        Returns:
            pd.DataFrame: 扩展市场日线数据
        """
        try:
            print(f"\n📈 读取扩展市场 {symbol} 日线数据...")
            
            data = self.ext_reader.daily(symbol=symbol)
            
            if data is not None and not data.empty:
                print(f"✓ 成功读取 {len(data)} 条扩展市场日线数据")
                return data
            else:
                print(f"✗ 未找到 {symbol} 的扩展市场数据")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ 读取扩展市场数据失败: {e}")
            return pd.DataFrame()
    
    def read_block_data(self, symbol='block_zs', group=False):
        """
        读取板块数据
        
        Args:
            symbol (str): 板块文件名，如 'block_zs', 'block_gn', 'block_fg'
            group (bool): 是否分组显示
            
        Returns:
            pd.DataFrame: 板块数据
        """
        try:
            print(f"\n📋 读取板块数据 {symbol}...")
            
            data = self.std_reader.block(symbol=symbol, group=group)
            
            if data is not None and not data.empty:
                print(f"✓ 成功读取板块数据，共 {len(data)} 条记录")
                return data
            else:
                print(f"✗ 未找到板块数据 {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ 读取板块数据失败: {e}")
            return pd.DataFrame()
    
    def export_data(self, data, filename, format_type='csv'):
        """
        导出数据到文件
        
        Args:
            data (pd.DataFrame): 要导出的数据
            filename (str): 文件名（不含扩展名）
            format_type (str): 文件格式 'csv', 'xlsx', 'json'
        """
        if data.empty:
            print("✗ 数据为空，无法导出")
            return
            
        try:
            # 创建输出目录
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # 构建完整文件路径
            full_filename = output_dir / f"{filename}.{format_type}"
            
            # 导出数据
            if format_type == 'csv':
                data.to_csv(full_filename, encoding='utf-8-sig', index=True)
            elif format_type == 'xlsx':
                data.to_excel(full_filename, index=True)
            elif format_type == 'json':
                data.to_json(full_filename, orient='index', date_format='iso', force_ascii=False, indent=2)
            else:
                print(f"✗ 不支持的文件格式: {format_type}")
                return
                
            print(f"✓ 数据已导出到: {full_filename}")
            
        except Exception as e:
            print(f"✗ 导出数据失败: {e}")
    
    def show_data_info(self, data, title="数据信息"):
        """显示数据基本信息"""
        if data.empty:
            print(f"{title}: 数据为空")
            return
            
        print(f"\n{title}:")
        print(f"  数据形状: {data.shape}")
        print(f"  列名: {list(data.columns)}")
        if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
            print(f"  时间范围: {data.index.min()} 至 {data.index.max()}")
        print(f"  前5行数据:")
        print(data.head())


def main():
    """主函数 - 演示各种数据读取功能"""
    
    print("=" * 60)
    print("通达信离线数据读取Demo")
    print("=" * 60)
    
    # 初始化读取器
    try:
        # 可以手动指定通达信目录，如：
        # reader = TdxOfflineReader(tdx_dir='C:/new_tdx')
        reader = TdxOfflineReader()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 测试股票代码列表
    test_symbols = ['600036', '000001', '688001']  # 招商银行、平安银行、科创板股票
    
    print("\n" + "=" * 60)
    print("1. 日线数据读取测试")
    print("=" * 60)
    
    for symbol in test_symbols:
        # 读取普通日线数据
        daily_data = reader.read_daily_data(symbol)
        if not daily_data.empty:
            reader.show_data_info(daily_data, f"{symbol} 日线数据")
            # 导出数据
            reader.export_data(daily_data, f"{symbol}_daily", 'csv')
            break  # 只演示第一个有数据的股票
    
    print("\n" + "=" * 60)
    print("2. 前复权日线数据读取测试")
    print("=" * 60)
    
    # 读取前复权数据
    for symbol in test_symbols:
        qfq_data = reader.read_daily_data(symbol, adjust='qfq')
        if not qfq_data.empty:
            reader.show_data_info(qfq_data, f"{symbol} 前复权日线数据")
            break
    
    print("\n" + "=" * 60)
    print("3. 分钟线数据读取测试")
    print("=" * 60)
    
    for symbol in test_symbols:
        # 读取1分钟数据
        minute1_data = reader.read_minute_data(symbol, suffix=1)
        if not minute1_data.empty:
            reader.show_data_info(minute1_data, f"{symbol} 1分钟线数据")
            break
        
        # 读取5分钟数据
        minute5_data = reader.read_minute_data(symbol, suffix=5)
        if not minute5_data.empty:
            reader.show_data_info(minute5_data, f"{symbol} 5分钟线数据")
            break
    
    print("\n" + "=" * 60)
    print("4. 5分钟线数据读取测试(fzline方法)")
    print("=" * 60)
    
    for symbol in test_symbols:
        fzline_data = reader.read_fzline_data(symbol)
        if not fzline_data.empty:
            reader.show_data_info(fzline_data, f"{symbol} 5分钟线数据(fzline)")
            break
    
    print("\n" + "=" * 60)
    print("5. 板块数据读取测试")
    print("=" * 60)
    
    # 读取板块数据
    block_types = ['block_zs', 'block_gn', 'block_fg']
    for block_type in block_types:
        block_data = reader.read_block_data(block_type)
        if not block_data.empty:
            reader.show_data_info(block_data, f"板块数据 {block_type}")
            reader.export_data(block_data, f"{block_type}", 'csv')
            break
    
    print("\n" + "=" * 60)
    print("6. 扩展市场数据读取测试")
    print("=" * 60)
    
    # 测试扩展市场数据（如果有的话）
    ext_symbols = ['4#CF7D0LAO', '29#A1801']
    for symbol in ext_symbols:
        ext_data = reader.read_ext_daily_data(symbol)
        if not ext_data.empty:
            reader.show_data_info(ext_data, f"扩展市场 {symbol} 数据")
            break
    
    print("\n" + "=" * 60)
    print("Demo运行完成！")
    print("=" * 60)
    print("说明:")
    print("- 如果某些数据读取失败，可能是因为本地没有对应的数据文件")
    print("- 请确保通达信客户端已下载相关数据")
    print("- 导出的文件保存在 'output' 目录中")
    print("- 更多功能请参考mootdx官方文档: https://www.mootdx.com")


if __name__ == '__main__':
    main()
