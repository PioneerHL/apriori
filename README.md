# Apriori关联规则挖掘算法

![Apriori Algorithm](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Association_rule_learning.svg/300px-Association_rule_learning.svg.png)

## 项目简介

本项目提供了Apriori关联规则挖掘算法的完整Python实现，专为科学研究设计。该算法能够从大规模数据集中挖掘出物品之间的关联关系，广泛应用于市场购物篮分析、推荐系统、医疗诊断、网络安全等领域的研究工作。

## 功能特点

- **完整的Apriori算法实现**：包括频繁项集挖掘和关联规则生成
- **多指标评估**：支持计算支持度(Support)、置信度(Confidence)和提升度(Lift)
- **结果可视化**：提供关联规则的图形化展示功能
- **结果导出**：支持将频繁项集和关联规则导出为CSV文件，便于后续分析
- **参数灵活调整**：可自定义最小支持度、最小置信度和最小提升度阈值
- **示例数据集**：包含示例购物篮数据，便于快速测试和理解算法
- **详细注释**：代码中包含详细的注释，方便研究人员理解和扩展

## 系统要求

- Python 3.6+ 环境
- 依赖库：
  - numpy
  - pandas
  - matplotlib

## 安装依赖

在运行代码前，请先安装必要的依赖库：

```bash
pip install numpy pandas matplotlib
```

## 使用方法

### 基本使用

1. 将数据集准备成列表的列表格式，每个子列表代表一次交易
2. 导入Apriori类并初始化
3. 调用fit方法训练模型
4. 使用各种方法获取结果

```python
from apriori_algorithm import Apriori

# 准备数据集
my_dataset = [
    ['商品A', '商品B', '商品C'],
    ['商品A', '商品B'],
    ['商品A', '商品C', '商品D'],
    # ... 更多交易记录
]

# 初始化算法，设置参数
apriori = Apriori(min_support=0.3, min_confidence=0.7, min_lift=1.0)

# 训练模型
apriori.fit(my_dataset)

# 获取关联规则
rules = apriori.get_rules(sort_by='lift', ascending=False)

# 获取频繁项集
itemsets = apriori.get_frequent_itemsets()
```

### 命令行直接运行示例

您也可以直接运行脚本查看示例结果：

```bash
python apriori_algorithm.py
```

### 结果导出

```python
# 导出关联规则到CSV文件
apriori.export_rules_to_csv('my_rules.csv')

# 导出频繁项集到CSV文件
apriori.export_frequent_itemsets_to_csv('my_itemsets.csv')

# 可视化规则并保存为图片
apriori.plot_rules(top_n=15)  # 显示前15条规则
```

## 输入输出格式

### 输入格式

算法接受的数据集格式为列表的列表，例如：

```python
dataset = [
    ['牛奶', '面包', '黄油'],          # 交易1
    ['牛奶', '面包', '尿布', '啤酒'],   # 交易2
    ['面包', '黄油', '尿布'],          # 交易3
    # ... 更多交易
]
```

### 输出格式

1. **关联规则**：以字典形式返回，包含以下字段：
   - `antecedent`：规则前件（条件项集）
   - `consequent`：规则后件（结果项集）
   - `support`：规则支持度
   - `confidence`：规则置信度
   - `lift`：规则提升度

2. **频繁项集**：以字典形式返回，键为项集，值为对应的支持度

## 评价指标解释

- **支持度(Support)**：表示项集在数据集中出现的频率。公式：Support(X→Y) = P(X∪Y)
- **置信度(Confidence)**：表示在购买了X的情况下购买Y的概率。公式：Confidence(X→Y) = P(Y|X)
- **提升度(Lift)**：表示购买了X后对购买Y的提升作用。公式：Lift(X→Y) = Confidence(X→Y)/P(Y)
  - Lift > 1：X和Y正相关
  - Lift = 1：X和Y无关
  - Lift < 1：X和Y负相关

## 参数调优建议

- **最小支持度(min_support)**：
  - 数据集较大时，可适当降低支持度阈值
  - 数据集较小时，应提高支持度阈值
  - 通常取值范围在0.1-0.3之间

- **最小置信度(min_confidence)**：
  - 根据业务需求调整，要求高可靠性时设置较高
  - 通常取值范围在0.5-0.8之间

- **最小提升度(min_lift)**：
  - 至少应大于1才有意义
  - 实际应用中可设置为1.2或更高

## 算法性能考虑

- Apriori算法的时间复杂度主要取决于数据集大小、项集数量和最小支持度阈值
- 对于大规模数据集，可以考虑：
  1. 增加最小支持度阈值
  2. 减少数据集中的项数
  3. 使用更高效的实现（如FP-Growth算法）

## 示例研究应用

1. **市场购物篮分析**：分析顾客购买行为，优化商品摆放和促销策略
2. **医疗诊断**：发现症状与疾病之间的关联关系
3. **网络安全**：识别异常访问模式，用于入侵检测
4. **推荐系统**：基于用户历史行为推荐相关物品
5. **社交媒体分析**：发现用户兴趣之间的关联

## 扩展与定制

如果您需要扩展或定制算法功能，可以考虑以下方向：

1. 实现并行计算以提高大规模数据处理能力
2. 添加更多评价指标（如杠杆率、确信度等）
3. 集成其他关联规则算法（如FP-Growth、Eclat等）进行比较研究
4. 开发交互式可视化界面

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 参考文献

1. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. In Proceedings of the 20th VLDB conference (pp. 487-499).
2. Han, J., Pei, J., & Kamber, M. (2011). Data mining: concepts and techniques (3rd ed.). Morgan Kaufmann.
3. Brin, S., Motwani, R., Ullman, J. D., & Tsur, S. (1997). Dynamic itemset counting and implication rules for market basket data. ACM SIGMOD Record, 26(2), 255-264.

## 使用提示

- 在进行科学研究时，建议对不同参数组合进行实验，选择最适合特定数据集的参数
- 导出的CSV文件可以使用Excel、Tableau等工具进行进一步分析和可视化
- 生成的规则需要结合领域知识进行解释和验证，不能仅依赖统计指标