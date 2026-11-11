# 成员二交付说明（v2：评分预测 / RMSE）

面向 **成员一（最终评分）**、**成员三（展示 Top-10）**、**成员四（报告）**。
工程细节看 `README.md`，这份只说「拿什么、怎么用、哪里会踩坑」。

> **v1（NDCG / Top-10 候选排序）已于 2026-11-10 作废**。相关代码与结果仍保留在仓库里作「推荐模块展示」补充实验，但**不能用于本次验收**。本文件全部是 v2 口径。

**仓库**：https://github.com/cxu7524-dotcom/MovieLens-Rec ｜ 当前提交 `1567970` ｜ 工作区干净 ｜ 36 个单测全绿

---

## 0. 三十秒摘要

**正式任务**：对 `test_pairs.csv` 的 100,000 个 (userId, movieId) 预测评分，用 **RMSE** 评价。

| 模型 | val RMSE | 相对 B0 | 状态 |
|---|---|---|---|
| random（自检下限） | 1.842889 | -76.0% | 评测器下限 |
| **global mean = B0（及格线）** | **1.047271** | 0 | 全队基线 |
| user mean | 1.111734 | -6.16% | 比 B0 还差，见坑 2 |
| item mean | 0.948220 | +9.46% | 物品偏差是主力 |
| **bias μ+b_u+b_i（λ=20）** | **0.929663** | **+11.23%** | 已过 10% 线 |
| **itemcf（K=200, β=20）** | **0.926035** | **+11.58%** | **主交付** |
| spark ALS（rank=20, iter=20, reg=0.1，全量） | 0.987469 | +5.71% | 分布式对照，精度不如 bias |
| lsh 近似邻居 | 0.929677 | +11.22% | 负结果，别用 |

- 交付模型 = **ItemCF（K=200, β=20）**：绝对下降 **0.121236**，相对下降 **11.58%**。
- 10% 目标线（我们的解释）：`RMSE ≤ 1.047271 × 0.9 = 0.942544`。**bias 与 itemcf 都过线。**
- 两半区（val）：MovieLens **0.885868** / 合成 **0.961735**。ItemCF 在合成半区**不如 bias**（0.9617 vs 0.9593）。
- **口径待老师确认**：①「提升 10%」是否就是 RMSE 降 10%；② RMSE 是否按用户加权（macro 与 micro 在 B0 上差 0.0199）。不阻塞交付，报告里写一句即可。

---

## 1. 文件清单：谁拿什么

### 给成员一 / 成员四（算最终 test RMSE）

| 文件 | 行数 | 说明 |
|---|---|---|
| **`results/pred_test.csv`** | **100,000** | **唯一交付文件**。`userId,movieId,pred_rating`，覆盖 `test_pairs.csv` 每一行、不多不少，已 clip 到 [0.5, 5.0]，实测范围 [0.5349, 5.0000] |

标签在老师/成员一处，本地算不出 test RMSE。拿到标签后：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\pred_test.csv --stage test --name pred_test
```

### 给成员四（写报告）

| 文件 | 内容 |
|---|---|
| `results/pred_val.csv` | val 预测 99,998 行，可与标签复算（复算结果 0.926035） |
| `results/v2_table_models.md` / `.csv` | 全模型 RMSE/MAE/相对提升 + 两半区，**报告主表直接用它** |
| `results/baselines_val.json` + `results/table_baselines.md` | 基线阶梯 5 级 + λ 扫描（1/5/10/20/50） |
| `results/v2_table_params.md` | ItemCF K×β 15 组、ALS 12 组网格、LSH 9 组网格 |
| `results/v2_table_efficiency.md` / `.csv` | 构建/预测耗时、峰值内存、吞吐 |
| `results/v2_table_scaling.md` / `.csv` + `results/scaling_curve_als.svg` | ALS 1/2/4/8 扩展性表与曲线 |
| `results/table_lsh.md` | 精确 vs LSH 的 overlap@K / RMSE / 耗时 |
| `results/delivery_summary.json` | 选型、B0、目标线、两半区、待确认口径，一份汇总 |
| `results/manifest_*.json` | 数据指纹、参数、种子、环境（复现用） |

### 给成员三（展示 Top-10）

- 排序键换成 `pred_rating`：`results/pred_val.csv` / `results/pred_test.csv` 第三列降序取前 10。
- **不要再接旧的 `scores_*.csv`**（三列 `score`，v1 口径，已作废）。
- 要排除已看：用训练集里该用户已评分的电影过滤（test 对本身不在训练集里，不会误删）。

---

## 2. 预测文件契约（照抄，别改）

```
userId,movieId,pred_rating
1,80,3.912441
```

- 有表头，三列，逗号分隔，行顺序无关。
- **必须覆盖该阶段每一对，一个不多一个不少**。评测器硬校验：重复对 / 缺对 / 多对 / NaN / 越界（不在 [0.5, 5.0]）**直接抛错**。
- **必须 clip 到 [0.5, 5.0]**，这是白捡的 RMSE 改善。

---

## 3. 五个坑（按被问到的概率排序）

1. **别按 `hidden_cutoff_timestamp` 过滤训练集。** 它是全局截断，训练集里有 31,806 条晚于它的**合法**记录（属于其他用户）。按它过滤会丢数据。
2. **user mean 比全局均值更差**（-6.16%）。测试用户是重活跃用户，个人均值被长尾低分拉偏。任何「按用户归一化」的想法先看这条。
3. **LSH 那套别进融合。** overlap@100 只有 0.91%，RMSE 退回 bias 水平，加上签名构建还更慢（0.68×）。
4. **val 与 test 的分半区对数不是同一批**：val 48,078 / 51,920，test 49,953 / 50,047。README 第 9 节已分开列，别抄错。
5. **扩展性表是 20% 样本（6,357,785 条）跑的**，所以表里 RMSE 是 1.22~1.28，不是全量的 0.987469。加速比与吞吐可比，**精度不能跨表比**。

---

## 4. 评测口径（全队唯一）

```powershell
# val（本地有标签）
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\pred_val.csv --stage val --name pred_val

# test（只校验形状与覆盖，RMSE 需标签）
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\pred_test.csv --stage test --name pred_test
```

- `RMSE = sqrt(mean((pred − truth)^2))`，**pair 等权（micro）**，与老师描述一致；同时输出 `rmse_user_weighted`（macro）备查。
- 两半区按 `id_ranges.json` 的电影 ID 分界 **87,585** 拆开报。
- 输出 `results/metrics_<name>.json`：`n_pairs / n_users / rmse / mae / by_half / pred_min / pred_max`。

---

## 5. 给成员四：报告里能直接用的五条结论

1. **物品偏差贡献了几乎全部收益。** item mean 一个人拿到 9.46%；完整 bias 到 11.23%。`b_u` 的边际很小（user mean 单独用还倒退 6.16%）。
2. **ItemCF 只比 bias 好 0.39%（0.003628）。** 余弦用二值共现，与评分残差相关性弱；K 从 50 加到 200 只多 0.0004。**诚实写：相似度在这个数据上边际收益有限，主力是偏置项。**
3. **LSH 是负结果，且是干净的负结果。** 网格内最好配置 overlap@100 = 0.91%，RMSE 退化到 bias，签名构建 139.9 s + 生成 100.0 s，比精确近邻慢。原因：142,609 部电影里中位评分条数只有 4，Jaccard 中位数极低，banding 撞桶概率约 `s^rows`，rows≥4 时对低 Jaccard 对几乎为 0。**建议措辞：我们量化了 MinHash-LSH 在本数据的适用边界，而不是「LSH 没用」。**
4. **ALS 的价值在吞吐不在精度。** 全量 31.8 M 条训练只用 **113.7 s（279,768 条/秒）**，但 val RMSE 0.987469，输给 bias。原因：调参在 10% 样本上做，最优 rank 只有 20，容量被样本卡住；显式 ALS 不利用均值信号；合成半区它更差（1.0031 vs bias 0.9593）。
5. **合成半区倒挂是过拟合 ML 分布的证据。** ItemCF 在 ML 半区 0.8859（优于 bias 0.8966），在合成半区 0.9617（差于 bias 0.9593）。这个对比可以单独写一段。

扩展性结论（可直接引用）：**8 核 3.79×，并行效率 0.47**；2 核效率最高 0.93，4 核起明显掉。ALS 每轮是两次全量 shuffle，写盘与序列化的固定开销不随核数下降，driver GC 也是固定成本。

---

## 6. 复现

```powershell
cd D:\movie-rec
$env:PYTHONPATH=""
.\.venv\Scripts\python.exe scripts\run_all.py              # v2 全流程，约 2 小时
.\.venv\Scripts\python.exe scripts\run_all.py --fast       # 轻量校验，约 1 分钟
.\.venv\Scripts\python.exe scripts\run_all.py --legacy     # 旧 v1 NDCG 展示链路
```

数据不在仓库里（`.gitignore` 排除 `data/`）。复现前需要把 `训练数据集` 放到 `data/hybrid/`，再跑 `scripts/verify_hybrid_data.py` 验收。单步命令见 `README.md` 第 2 节。

---

## 7. 已知不足（写进报告的「局限性」）

| 项 | 现状 |
|---|---|
| ItemCF 相对 bias 的增益 | 只有 +0.39%，相似度信号弱 |
| ALS 精度 | 0.987469，不如 bias；调参样本只有 10% |
| LSH | 负结果，无正收益配置 |
| tags（1.99 M 条） | 未使用，留给后续特征 |
| 扩展性实验 | 20% 样本，非全量 |
| 口径 | 「提升 10%」与 RMSE 加权方式待老师确认 |
