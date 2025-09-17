#!/bin/bash

# Apriori算法启动脚本
# 用于快速安装依赖并运行Apriori算法示例

# 设置脚本为遇到错误时退出
set -e

# 定义颜色变量
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 欢迎信息
welcome_message() {
    echo -e "${GREEN}===== Apriori关联规则挖掘算法 =====${NC}"
    echo -e "该脚本将帮助您快速运行Apriori算法示例。"
    echo -e "该算法可用于科学研究中的关联规则挖掘。"
    echo ""
}

# 检查Python环境
check_python() {
    echo -e "${YELLOW}正在检查Python环境...${NC}"
    
    # 检查Python3是否安装
    if command -v python3 &> /dev/null; then
        PYTHON=python3
        echo -e "找到了Python 3"
    elif command -v python &> /dev/null; then
        # 检查是否为Python 3
        PYTHON_VERSION=$(python --version 2>&1 | grep -o 'Python 3\.[0-9]\+')
        if [[ -n "$PYTHON_VERSION" ]]; then
            PYTHON=python
            echo -e "找到了$PYTHON_VERSION"
        else
            echo -e "${RED}错误: 未找到Python 3环境。请先安装Python 3。${NC}"
            exit 1
        fi
    else
        echo -e "${RED}错误: 未找到Python环境。请先安装Python 3。${NC}"
        exit 1
    fi
    
    # 检查pip是否可用
    if ! $PYTHON -m pip --version &> /dev/null; then
        echo -e "${RED}错误: pip不可用。请先安装pip。${NC}"
        exit 1
    fi
    
    echo ""
}

# 安装依赖
install_dependencies() {
    echo -e "${YELLOW}正在安装必要的依赖库...${NC}"
    
    # 安装numpy, pandas, matplotlib
    $PYTHON -m pip install --upgrade pip
    $PYTHON -m pip install numpy pandas matplotlib
    
    echo -e "${GREEN}依赖库安装完成！${NC}"
    echo ""
}

# 运行主程序示例
run_main_example() {
    echo -e "${YELLOW}正在运行Apriori算法主程序示例...${NC}"
    
    # 确保脚本有执行权限
    chmod +x apriori_algorithm.py
    
    # 运行主程序
    $PYTHON apriori_algorithm.py
    
    echo -e "${GREEN}主程序示例运行完成！${NC}"
    echo ""
}

# 运行测试脚本
run_tests() {
    echo -e "${YELLOW}是否运行完整的测试脚本？这将测试不同参数配置下的算法性能。(y/n): ${NC}"
    read -r run_test_choice
    
    if [[ "$run_test_choice" == "y" || "$run_test_choice" == "Y" ]]; then
        echo -e "${YELLOW}正在运行Apriori算法测试脚本...${NC}"
        
        # 确保脚本有执行权限
        chmod +x test_apriori.py
        
        # 运行测试脚本
        $PYTHON test_apriori.py
        
        echo -e "${GREEN}测试脚本运行完成！测试结果已保存为图表文件。${NC}"
    fi
    echo ""
}

# 显示使用说明
show_usage() {
    echo -e "${YELLOW}使用说明:${NC}"
    echo "1. 主程序 (apriori_algorithm.py):"
    echo "   - 包含完整的Apriori算法实现"
    echo "   - 可以直接导入到您的Python代码中使用"
    echo "   - 也可以直接运行查看示例结果"
    echo ""
    echo "2. 测试脚本 (test_apriori.py):"
    echo "   - 用于测试不同参数配置下的算法性能"
    echo "   - 包含算法正确性验证功能"
    echo "   - 生成各种性能比较图表"
    echo ""
    echo "3. 自定义使用:"
    echo "   - 准备您自己的数据集（列表的列表格式）"
    echo "   - 导入Apriori类并设置合适的参数"
    echo "   - 调用fit方法训练模型并获取结果"
    echo ""
    echo "4. 结果导出:"
    echo "   - 算法支持将关联规则和频繁项集导出为CSV文件"
    echo "   - 支持生成规则可视化图表"
    echo ""
    echo -e "${GREEN}详细使用说明请查看README.md文件。${NC}"
}

# 主函数
main() {
    welcome_message
    check_python
    install_dependencies
    run_main_example
    run_tests
    show_usage
    
    echo -e "${GREEN}===== 操作完成 =====${NC}"
}

# 执行主函数
main