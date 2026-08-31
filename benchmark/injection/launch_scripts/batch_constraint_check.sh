#!/bin/bash
# batch_constraint_check.sh
# SDC 约束检查工具的 Shell 包装脚本
# 用法: ./batch_constraint_check.sh [选项]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/batch_constraint_check.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -a, --all           运行所有检查 (默认使用推荐检查)"
    echo "  -s, --single <dir>  检查单个 Collector 目录"
    echo "  -o, --output <dir>  指定输出目录"
    echo "  --step <n>          指定检查的步数"
    echo ""
    echo "示例:"
    echo "  $0                              # 批量检查所有 test_db"
    echo "  $0 --all                        # 运行所有约束检查"
    echo "  $0 --single dtype_test_db/Collector  # 检查单个目录"
    echo ""
}

# 检查 Python 脚本是否存在
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo -e "${RED}错误: 找不到 Python 脚本 ${PYTHON_SCRIPT}${NC}"
    exit 1
fi

# 检查 duckdb 是否安装
if ! python -c "import duckdb" 2>/dev/null; then
    echo -e "${YELLOW}警告: duckdb 未安装，正在安装...${NC}"
    pip install duckdb
fi

# 解析参数
ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -a|--all)
            ARGS="${ARGS} --all"
            shift
            ;;
        -s|--single)
            ARGS="${ARGS} --single-db $2"
            shift 2
            ;;
        -o|--output)
            ARGS="${ARGS} --output-dir $2"
            shift 2
            ;;
        --step)
            ARGS="${ARGS} --step $2"
            shift 2
            ;;
        *)
            ARGS="${ARGS} $1"
            shift
            ;;
    esac
done

# 执行 Python 脚本
echo -e "${GREEN}启动 SDC 约束检查...${NC}"
python "${PYTHON_SCRIPT}" --megatron-root "${SCRIPT_DIR}" ${ARGS}

echo ""
echo -e "${GREEN}检查完成!${NC}"
