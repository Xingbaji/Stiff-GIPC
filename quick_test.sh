#!/bin/bash
#
# quick_test.sh - 快速重新编译并测试代码
#
# 用法:
#   ./quick_test.sh              # 默认: demo 1, 30帧
#   ./quick_test.sh -d 2         # 使用 demo 2
#   ./quick_test.sh -f 50        # 运行 50 帧
#   ./quick_test.sh -d 1 -f 100  # demo 1, 100帧
#   ./quick_test.sh -c           # 完全重新配置cmake
#   ./quick_test.sh -v           # 详细输出模式
#   ./quick_test.sh -h           # 显示帮助
#

set -e

# 默认参数
DEMO=1
FRAMES=30
CLEAN_BUILD=false
VERBOSE=false
TIMEOUT=300

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助
print_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --demo NUM      指定demo编号 (默认: 1)"
    echo "  -f, --frames NUM    指定运行帧数 (默认: 30)"
    echo "  -c, --clean         完全重新配置cmake"
    echo "  -t, --timeout NUM   超时时间(秒) (默认: 300)"
    echo "  -v, --verbose       详细输出模式"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                  # 默认运行 demo 1, 30帧"
    echo "  $0 -d 2 -f 100      # 运行 demo 2, 100帧"
    echo "  $0 -c               # 清理后重新编译"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--demo)
            DEMO="$2"
            shift 2
            ;;
        -f|--frames)
            FRAMES="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       Stiff-GIPC 快速编译测试工具${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 记录开始时间
TOTAL_START=$(date +%s)

# 1. 编译阶段
echo -e "${YELLOW}[1/2] 编译中...${NC}"
BUILD_START=$(date +%s)

mkdir -p build
cd build

# 如果需要清理或者CMakeCache不存在，则运行cmake
if [ "$CLEAN_BUILD" = true ] || [ ! -f "CMakeCache.txt" ]; then
    echo -e "${YELLOW}      配置 CMake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

# 编译
if [ "$VERBOSE" = true ]; then
    make -j$(nproc)
else
    make -j$(nproc) 2>&1 | tail -20
fi

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [ $? -eq 0 ]; then
    echo -e "${GREEN}      编译成功! (耗时: ${BUILD_TIME}秒)${NC}"
else
    echo -e "${RED}      编译失败!${NC}"
    exit 1
fi

# 2. 测试阶段
echo ""
echo -e "${YELLOW}[2/2] 运行测试...${NC}"
echo -e "      Demo: ${DEMO}, 帧数: ${FRAMES}, 超时: ${TIMEOUT}秒"
echo ""

RUN_START=$(date +%s)

# 运行模拟，使用timeout防止卡死
if [ "$VERBOSE" = true ]; then
    timeout $TIMEOUT ./gipc --demo $DEMO --headless --frames $FRAMES 2>&1
    RUN_STATUS=$?
else
    # 简洁输出模式：只显示关键信息
    timeout $TIMEOUT ./gipc --demo $DEMO --headless --frames $FRAMES 2>&1 | \
        grep -E "(Frame|Kappa:|iteration k:|average time|complete|error|Error|WARNING|linesearch)" | \
        head -100
    RUN_STATUS=${PIPESTATUS[0]}
fi

RUN_END=$(date +%s)
RUN_TIME=$((RUN_END - RUN_START))

echo ""
echo -e "${BLUE}============================================${NC}"

# 总结
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

if [ $RUN_STATUS -eq 0 ] || [ $RUN_STATUS -eq 134 ]; then
    # 134 是正常退出时CUDA cleanup的信号，可以忽略
    echo -e "${GREEN}✓ 测试完成!${NC}"
    echo -e "  编译时间: ${BUILD_TIME}秒"
    echo -e "  运行时间: ${RUN_TIME}秒"
    echo -e "  总耗时:   ${TOTAL_TIME}秒"
elif [ $RUN_STATUS -eq 124 ]; then
    echo -e "${YELLOW}⚠ 测试超时 (${TIMEOUT}秒)${NC}"
    echo -e "  编译时间: ${BUILD_TIME}秒"
else
    echo -e "${RED}✗ 测试失败! (退出码: $RUN_STATUS)${NC}"
    echo -e "  编译时间: ${BUILD_TIME}秒"
    exit $RUN_STATUS
fi

echo -e "${BLUE}============================================${NC}"

