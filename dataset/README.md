# 数据集目录

四个数据集子目录：**abide**、**adni**、**mdd**、**ucla**。

在每个子目录下放置被试的 .mat 文件，命名为 **sub1.mat**、**sub2.mat**、**sub3.mat** …（数字连续或任意，脚本会按数字排序处理）。

## .mat 文件内容

每个 .mat 文件应包含**时间序列**，形状为 **(时间点数 T, 脑区数 N)**。  
脚本会尝试读取以下键（按顺序）：`TC`、`timecourse`、`timeseries`、`data`、`ts`；若为 3 维则取第一个被试。  
若键名不同，脚本会尝试使用文件中最大的 2 维数值数组作为 (T, N)。

## 计算与输出

在项目根目录执行：

```bash
# 使用高斯近似计算 ucla
python compute.py --method gauss --dataset ucla

# 使用随机迹估计计算 adni
python compute.py --method random --dataset adni

# 对四个数据集 (abide adni mdd ucla) 全部用 gauss 计算
python compute.py --method gauss --dataset all
```

计算结果会保存到 **MvHo-IB++/computed/abide**、**computed/adni**、**computed/mdd**、**computed/ucla** 下，每个被试一个文件：**sub1.pt**、**sub2.pt** …，内含 3D O-information 张量及元信息。
