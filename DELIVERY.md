# 成员二交付说明（v2：评分预测 / RMSE）

面向 **成员一（数据+最终评分）**、**成员三（展示 Top-10）**、**成员四（报告）**。工程细节看 `README.md`；这份只说「拿什么、怎么用、哪里有坑」。

> **v1（NDCG / Top-10 候选排序）已于 2026-11-10 作废**，相关文件仍保留在仓库里作「推荐模块展示」补充实验，但**不能用于本次验收**。本文件全部是 v2 口径。

---

## 0. 三十秒摘要

**正式任务**：对 `test_pairs.csv` 的 100,000 个 (userId, movieId) 预测评分，用 RMSE 评价。

| 模型 | val RMSE | 相对 B0 | 状态 |
|---|---|---|---|
| random（自检） | 1.842889 | -76.0% | 下限 |
| **global mean = B0（及格线）** | **1.047271** | 0 | 基线 |
| user mean | 1.111734 | -6.16% | 比 B0 差 |
| item mean | 0.948220 | +9.46% | 参考 |
| **bias μ+b_u+b_i（λ=20）** | **0.929663** | **+11.23%** | 已过 10% 目标 |
| **itemcf（K=200, β=20）** | **0.926035** | **+11.58%** | **主交付** |
| lsh 近似邻居 | 0.929677 | +11.22% | 实验，别用 |
| spark ALS（rank=20, iter=20, reg=0.1） | 0.987469 | +5.71% | 分布式对照，精度不如 bias |

- 交付模型 = **ItemCF（K=200, β=20）**：val RMSE **0.926035**，绝对下降 0.121236，相对下降 **11.58%**。
- 10% 目标线（我们的解释）：`RMSE ≤ 1.047271 × 0.9 = 0.942544`。bias 与 itemcf 都过线。
- 两半区：MovieLens **0.885868** / 合成 **0.961735**。ItemCF 在合成半区**不如 bias**（0.9617 vs 0.9593），说明它偏向 ML 分布。
- **口径待老师确认**：10% 的定义、RMSE 是否按用户加权。两项都已在 README 第 10 节写明，不阻塞交付。

---

## 1. 文件清单：谁拿什么

### 给成员一 / 成员四（算最终 test RMSE）

| 文件 | 行数 | 说明 |
|---|---|---|
| **`results/pred_test.csv`** | **100,000** | **唯一交付文件**：`userId,movieId,pred_rating`，覆盖 `test_pairs.csv` 每一行、不多不少，已 clip 到 [0.5, 5.0]，实测范围 [0.5349, 5.0000] |

标签在老师/成员一处，本地算不出 test RMSE；拿到标签后直接用第 4 节的命令跑。

### 给成员四（写报告）

| 文件 | 内容 |
|---|---|
| `results/pred_val.csv` | val 预测（99,998 行），可与标签复算 |
| `results/baselines_val.json` + `results/table_baselines.md` | 基线阶梯 5 级 + λ 扫描 |
| `results/v2_table_models.md` / `.csv` | 全模型 val RMSE/MAE/相对提升 + 两半区 |
| `results/v2_table_efficiency.md` / `.csv` | 构建/预测耗时、峰值内存、吞吐 |
| `results/v2_table_params.md` | ItemCF K×β 扫描、ALS 12 组网格、LSH 9 组网格 |
| `results/table_lsh.md` | 精确 vs LSH 的 overlap@K / RMSE / 耗时 |
| `results/v2_table_scaling.md` / `.csv` + `results/scaling_curve_als.svg` | ALS 1/2/4/8 扩展性表与曲线 |
| `results/delivery_summary.json` | 选型、B0、目标线、两半区、待确认口径，一份汇总 |
| `results/manifest_*.json` | 数据指纹（size/mtime/sha1 前 4 MiB）、参数、种子、环境 |

### 给成员三（展示 Top-10）

以前按 `score` 排序，现在改成**按预测评分排序**：

- `results/pred_val.csv` / `results/pred_test.csv` 的第三列 `pred_rating` 就是排序键，降序取前 10 即可。
- **不要再接旧的 `scores_*.csv`**（三列 `score`，v1 口径，已作废）。
- 若要排除已看：用训练集里该用户已评分的电影做过滤（注意 test 对本身不在训练集里）。

---

## 2. 预测文件契约（照抄，别改）

```
userId,movieId,pred_rating
1,80,3.912441
```

- **有表头**，三列，逗号分隔。
- **必须覆盖该阶段每一对，一个不多一个不少**；评测器会硬校验：重复对 / 缺对 / 多对 / NaN / 越界（不在 [0.5, 5.0]）**直接报错**。
- **clip 到 [0.5, 5.0]**：这是白捡的 RMSE 改善，别省。
- 行顺序无关（评测器按 `(userId, movieId)` 对齐），但别丢行。

---

## 3. 四个必须知道的坑

1. **别按 `hidden_cutoff_timestamp` 过滤训练集。** 它是全局截断，训练集里有 31,806 条晚于它的**合法**记录（其他用户的）。按它过滤会丢数据。
2. **user mean 比全局均值更差**（-6.16%）。测试用户是重活跃用户，个人均值被长尾低分拉偏。任何「按用户归一化」的想法先看这条。
3. **LSH 那套别进融合**（overlap@100 0.91%，RMSE 退回 bias，还更慢）。
4. **test 无标签**，`pred_test.csv` 只能交给成员一/四验；本地只能验 val。

---

## 4. 评测口径（全队唯一）

```powershell
# val（本地有标签）
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\pred_val.csv --stage val --name pred_val

# test（只校验形状与覆盖，RMSE 需要标签）
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\pred_test.csv --stage test --name pred_test
```

- RMSE = `sqrt(mean((pred − truth)^2))`，**pair 等权（micro）**，与老师描述一致；同时输出 `rmse_user_weighted`（macro）备查。
- 两半区（MovieLens / 合成）按 `id_ranges.json` 的电影 ID 分界 87,585 拆开报。
- 输出 `results/metrics_<name>.json`，含 `n_pairs / n_users / rmse / mae / by_half / pred_min / pred_max`。

---

## 5. 复现

```powershell
cd D:\movie-rec
$env:PYTHONPATH=""
.\.venv\Scripts\python.exe scripts\run_all.py              # v2 全流程，约 2 小时
.\.venv\Scripts\python.exe scripts\run_all.py --fast       # 只跑轻量校验，约 1 分钟
.\.venv\Scripts\python.exe scripts\run_all.py --legacy     # 旧 v1 NDCG 展示链路
```

单步命令见 `README.md` 第 2 节。
