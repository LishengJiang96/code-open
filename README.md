# ELECTRE-T2B 代码

论文《An improved ELECTRE~III method integrating bootstrap-based thresholds and Bonferroni mean for urban sustainability assessment》的计算代码。

## 依赖

- Python 3.9+
- numpy, pandas, scipy

## 结构

```
chapter3/   方法实现与示例
  electre_t2b.py        核心计算（阈值、权重、一致度、可信度、综合指数）
  example_threshold.py  示例 1：阈值与权重
  example_bm.py         示例 2：交互参数
chapter4/   模拟验证
  threshold_robustness.py  4.1 阈值估计稳健性（4050 组合，B=5000，1000 次重复）
  rank_reversal.py         4.2 秩反转（972 组合，1000 次重复）
chapter5/   案例
  case_study.py     表 4、表 7
  sensitivity.py    表 8
  comparative.py    表 9
chapter6/   讨论
  threshold_analysis.py  6.2 阈值比
data/       案例数据 data_sc.xlsx
```

## 运行

在对应目录执行，如 `python chapter5/case_study.py`。

第 4 章模拟计算量大（4050/972 组合 × 1000 次重复），需较长时间；全部随机数使用固定种子 SEED=2026，结果可复现。
