#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apriori算法测试脚本
用于测试不同参数配置下的算法性能和结果

该脚本提供了多种测试用例：
1. 不同最小支持度参数的比较
2. 不同最小置信度参数的比较
3. 不同数据集大小的性能测试
4. 与标准结果的对比验证
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apriori_algorithm import Apriori

# 测试用数据集生成函数

def generate_synthetic_dataset(n_transactions, n_items, item_probability=0.3, seed=None):
    """
    生成合成交易数据集
    
    参数:
        n_transactions: 交易数量
        n_items: 物品总数
        item_probability: 每个物品出现在交易中的概率
        seed: 随机种子，保证结果可复现
    
    返回:
        生成的数据集（列表的列表）
    """
    if seed is not None:
        np.random.seed(seed)
    
    items = [f'物品{i}' for i in range(1, n_items + 1)]
    dataset = []
    
    for _ in range(n_transactions):
        transaction = []
        for item in items:
            if np.random.random() < item_probability:
                transaction.append(item)
        # 确保每个交易至少有一个物品
        if not transaction:
            transaction.append(np.random.choice(items))
        dataset.append(transaction)
    
    return dataset

# 测试函数

def test_min_support():"""
测试不同最小支持度参数对结果的影响
"""
    print("\n===== 测试不同最小支持度参数 =====")
    
    # 准备数据集
    dataset = generate_synthetic_dataset(100, 20, item_probability=0.25, seed=42)
    
    # 测试不同的支持度值
    support_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    results = []
    
    for support in support_values:
        print(f"\n测试支持度: {support}")
        apriori = Apriori(min_support=support, min_confidence=0.6, min_lift=1.0)
        
        # 记录执行时间
        start_time = time.time()
        apriori.fit(dataset)
        end_time = time.time()
        
        # 统计结果
        n_itemsets = sum(len(items) for items in apriori.frequent_itemsets.values())
        n_rules = len(apriori.rules)
        
        print(f"执行时间: {end_time - start_time:.4f}秒")
        print(f"发现频繁项集数量: {n_itemsets}")
        print(f"生成关联规则数量: {n_rules}")
        
        results.append({
            'support': support,
            'execution_time': end_time - start_time,
            'n_itemsets': n_itemsets,
            'n_rules': n_rules
        })
    
    # 可视化结果
    plot_support_comparison(results)
    
    return results

def test_min_confidence():
    """
    测试不同最小置信度参数对结果的影响
    """
    print("\n===== 测试不同最小置信度参数 =====")
    
    # 准备数据集
    dataset = generate_synthetic_dataset(100, 20, item_probability=0.25, seed=42)
    
    # 测试不同的置信度值
    confidence_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for confidence in confidence_values:
        print(f"\n测试置信度: {confidence}")
        apriori = Apriori(min_support=0.2, min_confidence=confidence, min_lift=1.0)
        
        # 记录执行时间
        start_time = time.time()
        apriori.fit(dataset)
        end_time = time.time()
        
        # 统计结果
        n_itemsets = sum(len(items) for items in apriori.frequent_itemsets.values())
        n_rules = len(apriori.rules)
        
        print(f"执行时间: {end_time - start_time:.4f}秒")
        print(f"发现频繁项集数量: {n_itemsets}")
        print(f"生成关联规则数量: {n_rules}")
        
        results.append({
            'confidence': confidence,
            'execution_time': end_time - start_time,
            'n_itemsets': n_itemsets,
            'n_rules': n_rules
        })
    
    # 可视化结果
    plot_confidence_comparison(results)
    
    return results

def test_dataset_size():
    """
    测试不同数据集大小对算法性能的影响
    """
    print("\n===== 测试不同数据集大小 =====")
    
    # 测试不同的数据集大小
    dataset_sizes = [100, 500, 1000, 2000, 5000]
    results = []
    
    for size in dataset_sizes:
        print(f"\n测试数据集大小: {size}")
        # 生成不同大小的数据集
        dataset = generate_synthetic_dataset(size, 20, item_probability=0.25, seed=42)
        
        apriori = Apriori(min_support=0.2, min_confidence=0.6, min_lift=1.0)
        
        # 记录执行时间
        start_time = time.time()
        apriori.fit(dataset)
        end_time = time.time()
        
        # 统计结果
        n_itemsets = sum(len(items) for items in apriori.frequent_itemsets.values())
        n_rules = len(apriori.rules)
        
        print(f"执行时间: {end_time - start_time:.4f}秒")
        print(f"发现频繁项集数量: {n_itemsets}")
        print(f"生成关联规则数量: {n_rules}")
        
        results.append({
            'dataset_size': size,
            'execution_time': end_time - start_time,
            'n_itemsets': n_itemsets,
            'n_rules': n_rules
        })
    
    # 可视化结果
    plot_dataset_size_comparison(results)
    
    return results

def validate_algorithm():
    """
    验证算法结果的正确性
    使用已知结果的小型数据集进行验证
    """
    print("\n===== 验证算法结果正确性 =====")
    
    # 小型测试数据集，已知预期结果
    test_dataset = [
        ['A', 'B', 'C'],
        ['A', 'B'],
        ['A', 'C'],
        ['B', 'C'],
        ['A', 'D']
    ]
    
    # 设置较小的支持度和置信度以获取所有可能的规则
    apriori = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
    apriori.fit(test_dataset)
    
    # 显示所有频繁项集和规则
    print("\n所有频繁项集:")
    for k, itemsets in apriori.frequent_itemsets.items():
        for itemset, support in itemsets.items():
            print(f"{set(itemset)}: 支持度 = {support:.4f}")
    
    print("\n所有关联规则:")
    rules = apriori.get_rules()
    for i, rule in enumerate(rules):
        print(f"\n规则 {i+1}:")
        print(f"前件: {rule['antecedent']}")
        print(f"后件: {rule['consequent']}")
        print(f"支持度: {rule['support']:.4f}")
        print(f"置信度: {rule['confidence']:.4f}")
        print(f"提升度: {rule['lift']:.4f}")
    
    # 计算预期的支持度和置信度以验证
    # 例如，项集{A,B}的支持度应为2/5=0.4
    # 规则A->B的置信度应为2/3≈0.6667
    validation_passed = True
    
    # 验证特定项集的支持度
    itemset_ab = frozenset(['A', 'B'])
    expected_support_ab = 2/5  # 数据集中有2条交易包含A和B
    actual_support_ab = None
    
    for itemsets in apriori.frequent_itemsets.values():
        if itemset_ab in itemsets:
            actual_support_ab = itemsets[itemset_ab]
            break
    
    if actual_support_ab is not None and abs(actual_support_ab - expected_support_ab) < 0.001:
        print("\n✓ 项集{A,B}支持度验证通过")
    else:
        print(f"\n✗ 项集{A,B}支持度验证失败: 预期={expected_support_ab}, 实际={actual_support_ab}")
        validation_passed = False
    
    # 验证特定规则的置信度
    # 寻找规则A->B
    rule_found = False
    for rule in rules:
        if rule['antecedent'] == {'A'} and rule['consequent'] == {'B'}:
            rule_found = True
            expected_confidence = 2/3  # 包含A的3条交易中有2条包含B
            if abs(rule['confidence'] - expected_confidence) < 0.001:
                print("✓ 规则A->B置信度验证通过")
            else:
                print(f"✗ 规则A->B置信度验证失败: 预期={expected_confidence}, 实际={rule['confidence']}")
                validation_passed = False
            break
    
    if not rule_found:
        print("✗ 规则A->B未找到")
        validation_passed = False
    
    return validation_passed

# 可视化函数

def plot_support_comparison(results):
    """
    可视化不同支持度参数下的结果比较
    """
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(df['support'], df['execution_time'], marker='o')
    axes[0].set_title('最小支持度对执行时间的影响')
    axes[0].set_xlabel('最小支持度')
    axes[0].set_ylabel('执行时间 (秒)')
    axes[0].grid(True)
    
    axes[1].plot(df['support'], df['n_itemsets'], marker='o', color='green')
    axes[1].set_title('最小支持度对频繁项集数量的影响')
    axes[1].set_xlabel('最小支持度')
    axes[1].set_ylabel('频繁项集数量')
    axes[1].grid(True)
    
    axes[2].plot(df['support'], df['n_rules'], marker='o', color='red')
    axes[2].set_title('最小支持度对关联规则数量的影响')
    axes[2].set_xlabel('最小支持度')
    axes[2].set_ylabel('关联规则数量')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('support_comparison.png', dpi=300)
    plt.close()
    print("支持度比较图已保存为 support_comparison.png")

def plot_confidence_comparison(results):
    """
    可视化不同置信度参数下的结果比较
    """
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 执行时间变化
    axes[0].plot(df['confidence'], df['execution_time'], marker='o')
    axes[0].set_title('最小置信度对执行时间的影响')
    axes[0].set_xlabel('最小置信度')
    axes[0].set_ylabel('执行时间 (秒)')
    axes[0].grid(True)
    
    # 关联规则数量变化
    axes[1].plot(df['confidence'], df['n_rules'], marker='o', color='red')
    axes[1].set_title('最小置信度对关联规则数量的影响')
    axes[1].set_xlabel('最小置信度')
    axes[1].set_ylabel('关联规则数量')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('confidence_comparison.png', dpi=300)
    plt.close()
    print("置信度比较图已保存为 confidence_comparison.png")

def plot_dataset_size_comparison(results):
    """
    可视化不同数据集大小下的性能比较
    """
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 执行时间变化（对数坐标）
    axes[0].plot(df['dataset_size'], df['execution_time'], marker='o')
    axes[0].set_xscale('log')
    axes[0].set_title('数据集大小对执行时间的影响')
    axes[0].set_xlabel('数据集大小 (交易数量)')
    axes[0].set_ylabel('执行时间 (秒)')
    axes[0].grid(True)
    
    # 频繁项集数量变化
    axes[1].plot(df['dataset_size'], df['n_itemsets'], marker='o', color='green')
    axes[1].set_title('数据集大小对频繁项集数量的影响')
    axes[1].set_xlabel('数据集大小 (交易数量)')
    axes[1].set_ylabel('频繁项集数量')
    axes[1].grid(True)
    
    # 关联规则数量变化
    axes[2].plot(df['dataset_size'], df['n_rules'], marker='o', color='red')
    axes[2].set_title('数据集大小对关联规则数量的影响')
    axes[2].set_xlabel('数据集大小 (交易数量)')
    axes[2].set_ylabel('关联规则数量')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('dataset_size_comparison.png', dpi=300)
    plt.close()
    print("数据集大小比较图已保存为 dataset_size_comparison.png")

# 主函数
if __name__ == "__main__":
    print("===== Apriori算法测试脚本 =====")
    print("本脚本将测试Apriori算法在不同参数和数据集下的性能和结果")
    
    # 运行所有测试
    support_results = test_min_support()
    confidence_results = test_min_confidence()
    size_results = test_dataset_size()
    
    # 验证算法正确性
    print("\n===== 算法正确性验证结果 =====")
    validation_result = validate_algorithm()
    if validation_result:
        print("\n✓ 所有验证通过！算法实现正确。")
    else:
        print("\n✗ 验证失败，请检查算法实现。")
    
    print("\n===== 测试完成 =====")
    print("所有测试结果已保存为图表文件，可在当前目录查看。")