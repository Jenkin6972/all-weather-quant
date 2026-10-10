#!/bin/bash

# 通达信离线数据读取Demo运行脚本
# ================================

echo "🚀 通达信离线数据读取Demo"
echo "=========================="

# 检查Python环境
echo "🔍 检查Python环境..."
python3 --version

# 检查mootdx是否已安装
echo "🔍 检查mootdx安装状态..."
python3 -c "import mootdx; print(f'mootdx版本: {mootdx.__version__}')" 2>/dev/null || {
    echo "❌ mootdx未安装，正在安装..."
    pip install 'mootdx[all]'
}

echo ""
echo "📋 可用的Demo:"
echo "1. simple_demo.py - 简单示例"
echo "2. demo_offline_reader.py - 完整功能演示"
echo "3. run_demo_docker.py - Docker环境专用"
echo ""

# 运行Docker专用Demo
echo "🐳 运行Docker环境专用Demo..."
python3 run_demo_docker.py

echo ""
echo "✅ Demo运行完成！"
echo ""
echo "📖 更多信息请查看:"
echo "- README_DEMO.md: 详细使用说明"
echo "- https://www.mootdx.com: 官方文档"
