# 通达信离线数据读取Demo使用说明

## 概述

本Demo演示如何使用mootdx库读取通达信客户端的离线数据，包括日线、分钟线、板块数据等。

## 文件说明

- `demo_offline_reader.py` - 完整功能演示
- `simple_demo.py` - 简化版本，适合快速上手
- `README_DEMO.md` - 本说明文件

## 环境要求

### 1. Python环境
- Python 3.8 或更高版本
- 推荐使用虚拟环境

### 2. 安装mootdx
```bash
# 安装完整版本（推荐）
pip install 'mootdx[all]'

# 或者安装基础版本
pip install mootdx
```

### 3. 通达信客户端
- 需要安装通达信客户端
- 确保已下载相关股票数据
- 记住安装目录路径

## 通达信数据目录结构

```
C:/new_tdx/                    # 通达信安装目录
└── vipdoc/                    # 数据目录
    ├── sh/                    # 上海市场
    │   ├── lday/             # 日线数据 (*.day)
    │   ├── minline/          # 1分钟数据 (*.1, *.lc1)
    │   └── fzline/           # 5分钟数据 (*.5, *.lc5)
    ├── sz/                    # 深圳市场
    │   ├── lday/             # 日线数据
    │   ├── minline/          # 分钟数据
    │   └── fzline/           # 5分钟数据
    └── ds/                    # 扩展市场(期货等)
        └── lday/             # 日线数据
```

## 快速开始

### 1. 运行简单示例
```bash
python simple_demo.py
```

### 2. 运行完整示例
```bash
python demo_offline_reader.py
```

### 3. 修改配置
在代码中修改通达信安装目录：
```python
# 修改为你的通达信安装目录
tdx_dir = 'C:/new_tdx'  # Windows
# tdx_dir = '/path/to/tdx'  # Linux/Mac
```

## 主要功能

### 1. 日线数据读取
```python
from mootdx.reader import Reader

reader = Reader.factory(market='std', tdxdir='C:/new_tdx')

# 读取普通日线数据
daily_data = reader.daily(symbol='600036')

# 读取前复权数据
qfq_data = reader.daily(symbol='600036', adjust='qfq')

# 读取后复权数据
hfq_data = reader.daily(symbol='600036', adjust='hfq')
```

### 2. 分钟线数据读取
```python
# 读取1分钟数据
minute1_data = reader.minute(symbol='600036', suffix=1)

# 读取5分钟数据
minute5_data = reader.minute(symbol='600036', suffix=5)

# 或者使用fzline方法读取5分钟数据
fzline_data = reader.fzline(symbol='600036')
```

### 3. 扩展市场数据
```python
# 创建扩展市场读取器
ext_reader = Reader.factory(market='ext', tdxdir='C:/new_tdx')

# 读取期货等扩展市场数据
ext_data = ext_reader.daily(symbol='29#A1801')
```

### 4. 板块数据读取
```python
# 读取板块数据
block_data = reader.block(symbol='block_zs')  # 指数板块
block_data = reader.block(symbol='block_gn')  # 概念板块
block_data = reader.block(symbol='block_fg')  # 风格板块
```

## 股票代码规则

### 上海市场 (sh)
- 60xxxx: 主板股票
- 68xxxx: 科创板
- 50xxxx, 51xxxx: ETF基金
- 11xxxx: 债券

### 深圳市场 (sz)
- 00xxxx: 主板股票
- 30xxxx: 创业板
- 12xxxx: 债券
- 15xxxx, 16xxxx: ETF基金

### 代码示例
- 600036: 招商银行 (上海)
- 000001: 平安银行 (深圳)
- 688001: 华兴源创 (科创板)
- 300001: 特锐德 (创业板)

## 数据字段说明

### 日线数据字段
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `amount`: 成交额

### 分钟线数据字段
- `datetime`: 时间
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `amount`: 成交额

## 常见问题

### 1. 找不到数据文件
**问题**: 提示"tdxdir 目录不存在"或读取数据为空

**解决方案**:
- 检查通达信安装目录是否正确
- 确保vipdoc目录存在
- 检查是否已下载相关股票数据

### 2. 常见安装目录
```
Windows:
- C:/new_tdx
- C:/tdx
- D:/new_tdx
- D:/tdx

Linux/Mac (Wine):
- ~/.wine/drive_c/new_tdx
```

### 3. 数据更新
- 通达信客户端需要定期更新数据
- 建议每日开盘前更新数据
- 可以设置自动更新

### 4. 复权数据说明
- `qfq`: 前复权，以当前价格为基准向前调整
- `hfq`: 后复权，以上市价格为基准向后调整
- 不指定: 不复权，使用原始价格

## 性能优化

### 1. 批量读取
```python
symbols = ['600036', '000001', '600519']
data_dict = {}

for symbol in symbols:
    data_dict[symbol] = reader.daily(symbol=symbol)
```

### 2. 数据缓存
mootdx内置了缓存机制，重复读取相同数据会更快。

### 3. 内存管理
处理大量数据时注意内存使用，及时释放不需要的数据。

## 数据导出

### 支持的格式
- CSV: `data.to_csv('filename.csv')`
- Excel: `data.to_excel('filename.xlsx')`
- JSON: `data.to_json('filename.json')`
- HDF5: `data.to_hdf('filename.h5', 'data')`

### 导出示例
```python
# 导出为CSV（推荐）
daily_data.to_csv('600036_daily.csv', encoding='utf-8-sig')

# 导出为Excel
daily_data.to_excel('600036_daily.xlsx', index=True)
```

## 进阶用法

### 1. 自定义板块
```python
# 创建自定义板块
reader.block_new(name='我的股票池', symbol=['600036', '000001', '600519'])

# 读取自定义板块
my_stocks = reader.block_new()
```

### 2. 数据分析示例
```python
import matplotlib.pyplot as plt

# 读取数据
data = reader.daily(symbol='600036')

# 计算移动平均线
data['ma5'] = data['close'].rolling(5).mean()
data['ma20'] = data['close'].rolling(20).mean()

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['close'], label='收盘价')
plt.plot(data.index, data['ma5'], label='5日均线')
plt.plot(data.index, data['ma20'], label='20日均线')
plt.legend()
plt.title('600036 股价走势')
plt.show()
```

## 联系方式

- 项目主页: https://github.com/mootdx/mootdx
- 在线文档: https://www.mootdx.com
- 问题反馈: https://github.com/mootdx/mootdx/issues

## 免责声明

本项目仅供学习交流使用，不得用于任何商业目的。使用本项目获取的数据请遵守相关法律法规。
