# -*- coding: utf-8 -*-
"""
Apriori关联规则挖掘算法实现
用于科学研究的数据挖掘和关联规则分析

该实现包含Apriori算法的核心功能：
- 频繁项集挖掘
- 关联规则生成
- 支持度、置信度、提升度计算
- 可视化功能
- 从Excel文件读取数据的能力
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import time
import os

class Apriori:
    """Apriori关联规则挖掘算法类"""
    
    def __init__(self, min_support=0.2, min_confidence=0.7, min_lift=1.0):
        """
        初始化Apriori算法参数
        
        参数:
            min_support: 最小支持度阈值
            min_confidence: 最小置信度阈值
            min_lift: 最小提升度阈值
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.frequent_itemsets = {}
        self.rules = []
        self.dataset_size = 0
    
    def calculate_support(self, dataset, itemset):
        """
        计算项集的支持度
        
        参数:
            dataset: 交易数据集
            itemset: 要计算支持度的项集
        
        返回:
            支持度值
        """
        count = 0
        for transaction in dataset:
            if itemset.issubset(transaction):
                count += 1
        return count / self.dataset_size
    
    def generate_candidates(self, itemsets, k):
        """
        生成k项候选集
        
        参数:
            itemsets: 频繁(k-1)项集
            k: 项集长度
        
        返回:
            候选k项集
        """
        candidates = set()
        itemsets_list = list(itemsets.keys()) if isinstance(itemsets, dict) else list(itemsets)
        
        # 连接操作：合并两个k-1项集生成k项集
        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                item1 = itemsets_list[i]
                item2 = itemsets_list[j]
                
                # 合并两个项集
                merged = set(item1).union(set(item2))
                if len(merged) == k:
                    # 生成所有可能的排序以避免重复
                    item_list = sorted(merged)
                    candidates.add(frozenset(item_list))
        
        return candidates
    
    def fit(self, dataset):
        """
        训练Apriori模型, 挖掘频繁项集和关联规则
        
        参数:
            dataset: 交易数据集, 格式为列表的列表
        """
        start_time = time.time()
        
        # 1. 生成所有1-项频繁集
        items = set()
        for transaction in dataset:
            for item in transaction:
                items.add(frozenset([item]))
        
        self.dataset_size = len(dataset)
        frequent_1_itemsets = {}
        for item in items:
            support = self.calculate_support(dataset, item)
            if support >= self.min_support:
                frequent_1_itemsets[item] = support
        
        self.frequent_itemsets[1] = frequent_1_itemsets
        
        # 2. 迭代生成k-项频繁集
        k = 2
        while True:
            # 生成候选k项集
            candidates = self.generate_candidates(self.frequent_itemsets[k-1], k)
            if not candidates:
                break
            
            # 筛选频繁k项集
            frequent_k_itemsets = {}
            for candidate in candidates:
                support = self.calculate_support(dataset, candidate)
                if support >= self.min_support:
                    frequent_k_itemsets[candidate] = support
            
            if not frequent_k_itemsets:
                break
            
            self.frequent_itemsets[k] = frequent_k_itemsets
            k += 1
        
        # 3. 生成关联规则
        self._generate_rules(dataset)
        
        end_time = time.time()
        print(f"算法执行时间: {end_time - start_time:.4f} 秒")
    
    def _generate_rules(self, dataset):
        """
        基于频繁项集生成关联规则
        
        参数:
            dataset: 交易数据集
        """
        # 从2-项频繁集开始生成规则
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset in self.frequent_itemsets[k]:
                # 生成所有可能的非空真子集作为前件
                for i in range(1, k):
                    antecedents = combinations(itemset, i)
                    for antecedent in antecedents:
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # 计算置信度
                        support_itemset = self.frequent_itemsets[k][itemset]
                        support_antecedent = self.calculate_support(dataset, antecedent)
                        confidence = support_itemset / support_antecedent
                        
                        # 计算提升度
                        support_consequent = self.calculate_support(dataset, consequent)
                        lift = confidence / support_consequent if support_consequent > 0 else 0
                        
                        # 检查是否满足规则条件
                        if confidence >= self.min_confidence and lift >= self.min_lift:
                            self.rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support_itemset,
                                'confidence': confidence,
                                'lift': lift
                            })
    
    def get_rules(self, sort_by='lift', ascending=False):
        """
        获取生成的关联规则
        
        参数:
            sort_by: 排序字段, 可选'support', 'confidence', 'lift'
            ascending: 是否升序排列
        
        返回:
            规则列表(已排序)
        """
        return sorted(self.rules, key=lambda x: x[sort_by], reverse=not ascending)
    
    def get_frequent_itemsets(self, k=None):
        """
        获取频繁项集
        
        参数:
            k: 指定项集的长度, None表示获取所有
        
        返回:
            频繁项集字典
        """
        if k is not None:
            return {k: self.frequent_itemsets.get(k, {})}
        return self.frequent_itemsets
    
    def plot_rules(self, top_n=10, show_plot=True):
        """
        可视化关联规则的支持度, 置信度和提升度
        
        参数:
            top_n: 显示前N条规则
            show_plot: 是否直接显示图形(默认为True)
        """
        if not self.rules:
            print("没有发现满足条件的关联规则")
            return
            
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 获取前N条规则
        rules = self.get_rules(sort_by='lift', ascending=False)[:top_n]
        
        # 准备绘图数据
        rule_labels = []
        support_values = []
        confidence_values = []
        lift_values = []
        
        for rule in rules:
            antecedent_str = ', '.join(rule['antecedent'])
            consequent_str = ', '.join(rule['consequent'])
            rule_labels.append(f"{antecedent_str} → {consequent_str}")
            support_values.append(rule['support'])
            confidence_values.append(rule['confidence'])
            lift_values.append(rule['lift'])
        
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 支持度条形图
        axes[0].barh(rule_labels, support_values, color='skyblue')
        axes[0].set_title('关联规则支持度')
        axes[0].set_ylabel('规则')
        axes[0].set_xlabel('支持度')
        axes[0].set_xlim(0, 1)
        
        # 置信度条形图
        axes[1].barh(rule_labels, confidence_values, color='lightgreen')
        axes[1].set_title('关联规则置信度')
        axes[1].set_ylabel('规则')
        axes[1].set_xlabel('置信度')
        axes[1].set_xlim(0, 1)
        
        # 提升度条形图
        max_lift = max(lift_values) if lift_values else 1
        axes[2].barh(rule_labels, lift_values, color='lightcoral')
        axes[2].set_title('关联规则提升度')
        axes[2].set_ylabel('规则')
        axes[2].set_xlabel('提升度')
        axes[2].set_xlim(0, max_lift * 1.1)  # 设置合适的x轴范围
        
        # 添加参考线
        axes[2].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        
        # 添加数据标签
        for i, v in enumerate(support_values):
            axes[0].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        for i, v in enumerate(confidence_values):
            axes[1].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        for i, v in enumerate(lift_values):
            axes[2].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('association_rules_visualization.png', dpi=300, bbox_inches='tight')
        
        # 如果需要显示图形
        if show_plot:
            plt.show()
    
    def export_rules_to_csv(self, filename='association_rules.csv'):
        """
        将关联规则导出到CSV文件
        
        参数:
            filename: 输出文件名
        """
        if not self.rules:
            print("没有发现满足条件的关联规则")
            return
        
        # 准备导出数据
        data = []
        for rule in self.rules:
            data.append({
                'Antecedent': ', '.join(rule['antecedent']),
                'Consequent': ', '.join(rule['consequent']),
                'Support': rule['support'],
                'Confidence': rule['confidence'],
                'Lift': rule['lift']
            })
        
        # 创建DataFrame并导出
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"关联规则已导出到 {filename}")
    
    def export_frequent_itemsets_to_csv(self, filename='frequent_itemsets.csv'):
        """
        将频繁项集导出到CSV文件
        
        参数:
            filename: 输出文件名
        """
        if not self.frequent_itemsets:
            print("没有发现满足条件的频繁项集")
            return
        
        # 准备导出数据
        data = []
        for k, itemsets in self.frequent_itemsets.items():
            for itemset, support in itemsets.items():
                data.append({
                    'Itemset': ', '.join(itemset),
                    'Size': k,
                    'Support': support
                })
        
        # 创建DataFrame并导出
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"频繁项集已导出到 {filename}")

# 从Excel文件读取数据并执行Apriori算法
if __name__ == "__main__":
    print("=== Apriori关联规则挖掘分析 ===")
    
    try:
        # 读取Excel文件中的数据
        print("正在读取data.xlsx文件...")
        # 假设Excel文件有一列'items'，包含每个交易的商品列表，用逗号分隔
        # 尝试不同的引擎来读取Excel文件
        try:
            df = pd.read_excel('data.xlsx', engine='openpyxl')
        except Exception:
            try:
                df = pd.read_excel('data.xlsx', engine='xlrd')
            except Exception:
                # 如果两种引擎都失败，尝试创建一个简单的数据集作为示例
                print("警告：无法读取Excel文件，使用内置示例数据...")
                dataset = [
                    ['牛奶', '面包', '黄油'],
                    ['牛奶', '面包', '尿布', '啤酒'],
                    ['面包', '黄油', '尿布'],
                    ['牛奶', '面包', '黄油', '尿布'],
                    ['牛奶', '尿布', '啤酒'],
                    ['面包', '黄油'],
                    ['牛奶', '啤酒'],
                    ['牛奶', '面包', '黄油', '尿布', '啤酒'],
                    ['尿布', '啤酒'],
                    ['牛奶', '面包', '尿布']
                ]
                # 直接返回处理流程，跳过后续数据读取步骤
                print(f"使用示例数据：{len(dataset)} 条交易记录")
                
                # 初始化并训练模型
                print("\n初始化Apriori模型并开始训练...")
                apriori = Apriori(min_support=0.2, min_confidence=0.6, min_lift=1.0)
                apriori.fit(dataset)
                
                # 显示频繁项集
                print("\n发现的频繁项集:")
                for k, itemsets in apriori.frequent_itemsets.items():
                    print(f"\n{k}-项频繁集 ({len(itemsets)} 个):")
                    # 只显示前5个频繁项集
                    for i, (itemset, support) in enumerate(itemsets.items()):
                        if i < 5:
                            print(f"{set(itemset)}: 支持度 = {support:.4f}")
                        elif i == 5 and len(itemsets) > 5:
                            print(f"... 还有 {len(itemsets) - 5} 个频繁项集未显示")
                            break
                
                # 显示关联规则
                print("\n生成的关联规则:")
                rules = apriori.get_rules(sort_by='lift', ascending=False)
                # 显示前10条规则
                show_rules_count = min(10, len(rules))
                for i, rule in enumerate(rules[:show_rules_count]):
                    print(f"\n规则 {i+1}:")
                    print(f"前件: {rule['antecedent']}")
                    print(f"后件: {rule['consequent']}")
                    print(f"支持度: {rule['support']:.4f}")
                    print(f"置信度: {rule['confidence']:.4f}")
                    print(f"提升度: {rule['lift']:.4f}")
                
                # 导出结果
                print("\n导出分析结果...")
                apriori.export_rules_to_csv('association_rules.csv')
                apriori.export_frequent_itemsets_to_csv('frequent_itemsets.csv')
                
                # 可视化结果
                print("\n生成规则可视化...")
                apriori.plot_rules(top_n=10, show_plot=True)
                
                print("\n=== 分析完成 ===")
                print("结果文件已保存：")
                print("- association_rules.csv: 关联规则数据")
                print("- frequent_itemsets.csv: 频繁项集数据")
                print("- association_rules_visualization.png: 规则可视化图")
                print("\n提示: 要使用自己的Excel数据，请确保安装了openpyxl或xlrd库")
                print("可以通过命令安装: pip install openpyxl xlrd")
                exit(0)
        
        # 将数据转换为算法需要的格式：列表的列表
        dataset = []
        for _, row in df.iterrows():
            # 处理每一行数据，假设数据格式为逗号分隔的字符串
            if isinstance(row.iloc[0], str):
                items = [item.strip() for item in row.iloc[0].split(',')]
                dataset.append(items)
            elif isinstance(row.iloc[0], list):
                dataset.append(row.iloc[0])
            else:
                # 尝试其他格式的处理
                items = [str(item).strip() for item in row if pd.notna(item) and str(item).strip()]
                dataset.append(items)
        
        print(f"成功读取 {len(dataset)} 条交易记录")
        
        # 初始化并训练模型
        print("\n初始化Apriori模型并开始训练...")
        apriori = Apriori(min_support=0.2, min_confidence=0.6, min_lift=1.0)
        apriori.fit(dataset)
        
        # 显示频繁项集
        print("\n发现的频繁项集:")
        for k, itemsets in apriori.frequent_itemsets.items():
            print(f"\n{k}-项频繁集 ({len(itemsets)} 个):")
            # 只显示前5个频繁项集，避免输出过多
            for i, (itemset, support) in enumerate(itemsets.items()):
                if i < 5:
                    print(f"{set(itemset)}: 支持度 = {support:.4f}")
                elif i == 5 and len(itemsets) > 5:
                    print(f"... 还有 {len(itemsets) - 5} 个频繁项集未显示")
                    break
        
        # 显示关联规则
        print("\n生成的关联规则:")
        rules = apriori.get_rules(sort_by='lift', ascending=False)
        # 显示前10条规则
        show_rules_count = min(10, len(rules))
        for i, rule in enumerate(rules[:show_rules_count]):
            print(f"\n规则 {i+1}:")
            print(f"前件: {rule['antecedent']}")
            print(f"后件: {rule['consequent']}")
            print(f"支持度: {rule['support']:.4f}")
            print(f"置信度: {rule['confidence']:.4f}")
            print(f"提升度: {rule['lift']:.4f}")
        
        # 导出结果
        print("\n导出分析结果...")
        apriori.export_rules_to_csv('association_rules.csv')
        apriori.export_frequent_itemsets_to_csv('frequent_itemsets.csv')
        
        # 可视化结果
        print("\n生成规则可视化...")
        apriori.plot_rules(top_n=10, show_plot=True)
        
        print("\n=== 分析完成 ===")
        print("结果文件已保存：")
        print("- association_rules.csv: 关联规则数据")
        print("- frequent_itemsets.csv: 频繁项集数据")
        print("- association_rules_visualization.png: 规则可视化图")
        
    except FileNotFoundError:
        print("错误: 未找到data.xlsx文件!")
        print("请确保在当前目录下有一个名为data.xlsx的Excel文件，包含交易数据。")
        print("Excel文件格式建议：每行代表一个交易，包含购买的商品（可以是逗号分隔的字符串或多个列）")
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        print("请检查Excel文件格式是否正确。")