# 成员二交付说明

面向 **成员三（排序/集成）** 与 **成员四（评测/报告）**。工程细节看 `README.md`，这份只说「拿什么、怎么用、哪里有坑」。

---

## 0. 三十秒摘要

| 模型 | NDCG@10 | Recall@10 | 覆盖率 | 状态 |
|---|---|---|---|---|
| random（自检下限） | 0.0876 | 0.0935 | 1.0000 | 参考 |
| **popular = B0（及格线）** | **0.8752** | **0.8639** | 0.1637 | 基线 |
| **itemcf 精确（K=200）** | **0.9319** | **0.8756** | 0.3312 | **主交付** |
| itemcf Spark（local[16]） | 0.9319 | 0.8756 | 0.3312 | 与单机完全一致 |
| LSH（交付版 bands=64, rows=1） | 0.9133 | 0.8612 | 0.2583 | 仅作实验，别融合 |

- itemcf 相对 B0：**NDCG +0.0567（+6.49%）**，Recall +0.0117（+1.35%）。
- 任务书要求 +10% → 目标 NDCG **0.9627**，还差 **0.0308**。**这个差距不是靠调参能补的，见第 5 节。**

---

## 1. 文件清单：谁拿什么

### 给成员三（排序/集成）

| 文件 | 行数 | 大小 | 用途 |
|---|---|---|---|
| `results/scores_itemcf_val.csv` | 20,281,548 | 467 MB | val 阶段召回分（主） |
| `results/scores_itemcf_test.csv` | 20,660,431 | 476 MB | test 阶段召回分（主） |
| `results/scores_popular_val.csv` | 20,281,548 | 467 MB | B0 对照 + 融合兜底路 |
| `results/scores_lsh_val.csv` | 20,281,548 | 467 MB | **仅供对比，不建议进融合** |
| `results/scores_lsh_test.csv` | 20,660,431 | 476 MB | 同上 |
| `results/scores_spark_itemcf_test.csv` | 20,660,431 | 476 MB | 与 `scores_itemcf_test.csv` **内容等价**，二选一 |

> 合计约 2.8 GB，微信传不了。建议转 parquet（约 1/4 大小）：
>
> ```powershell
> .\.venv\Scripts\python.exe -c "import pandas as pd,glob;[pd.read_csv(f).to_parquet(f.replace('.csv','.parquet'),index=False,compression='zstd') for f in glob.glob('results/scores_*.csv')]"
> ```
>
> 或者不给文件、只给脚本让对方自己跑（`scripts/run_all.py --only itemcf`，2.5 分钟）。

### 给成员四（评测/报告）

| 文件 | 内容 |
|---|---|
| `results/table_quality.md` / `.csv` | 近似质量表：26 行，含 13 组 LSH 参数 |
| `results/table_efficiency.md` / `.csv` | 效率对比：耗时、峰值内存、单用户延迟、吞吐 |
| `results/table_scaling.md` / `.csv` | 扩展性：并行度 1/2/4/8/16 |
| `results/scaling_curve.svg` | 扩展性曲线（左：分阶段耗时；右：加速比 + 1.0× 参考线） |
| `results/lsh_val.json` | 13 组参数原始数据 + overlap + 签名耗时 |
| `results/itemcf_val.json` | K 扫描、相似度构建耗时、峰值内存 |
| `results/spark_scaling_val.json` | 扩展性原始数据 |
| `results/manifest_*.json` | 数据指纹（sha1 前 4 MiB）、参数、种子、环境，复现用 |

---

## 2. 分数文件契约（照抄，别改）

```
userId,movieId,score
1,298,0.233243
```

- **有表头**，三列，逗号分隔。
- **覆盖候选集每一对**，一个不多一个不少（已校验 `missing_pairs=0 / extra_pairs=0`）。
- 缺失或算不出的对**填 0，不要丢行**——丢行会让排序错位。
- 排序规则：同一 `userId` 内按 `score` 降序，**同分按候选文件原始顺序**（评测器就是这么断的，保持一致才能复现）。

---

## 3. 四个必须知道的坑

1. **分数跨文件不可比。** itemcf 是余弦累加（0 ~ 数十），popular / lsh 归一化到 [0,1]。融合前**各自归一化**（min-max 或按用户内最大值），别直接相加。
2. **LSH 那两列不要当召回源用。** 见第 6 节，它在题目要求的参数网格内连 B0 都打不过。
3. **test 阶段没有标签。** `scores_*_test.csv` 只能由成员一验；你手上只能验 val。
4. **别自己重算 NDCG。** 用 `scripts/evaluate.py`，口径写死在 `conf/base.toml`。口径不一致，四个人的数字永远对不上。

---

## 4. 评测口径（全队唯一）

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --scores results\scores_itemcf_val.csv --stage val --name itemcf_val
```

- 正样本：`validation.csv` 中 `rating >= 4.0`。
- NDCG@10：gain=1，discount=`1/log2(rank+1)`，IDCG 用**该用户在候选集内**的正样本数。
- Recall@10：命中数 / 该用户在候选集内的正样本数。
- 覆盖率：所有人 top-10 出现过的电影数 / 候选集内电影数（87,585）。
- 输出 `results/metrics_<name>.json`，含 `hits / positives_in_candidates / missing_pairs`，可自检。

**已独立复算验证**：抽 3,135 用户用朴素 Python 重写一遍，NDCG 0.9295 / Recall 0.8710，与评测器 0.9319 / 0.8756 一致（差异来自抽样）。评测器可信。

---

## 5. 关键发现：+10% 到底有多难（这条最重要）

### 5.1 Recall 上 +10% 是数学不可能的

候选集构造是「该阶段正样本 + 100 负样本」，用户正样本数中位数 4、均值 7.83、最大 552，**20.9% 的用户正样本超过 10 个**。

因此 Recall@10 的理论上限是：

```
mean over users of  min(10, npos) / npos  =  0.9110
```

而 +10% 要求 **0.9503 > 0.9110**。**无论模型多强，Recall@10 都到不了。** 报告里不要再拿 Recall 说「离目标差多少」，会被问死。

### 5.2 NDCG 的差距集中在「只有 1~2 个正样本」的用户

按候选内正样本数分桶（itemcf，val）：

| 桶（npos） | 用户数 | 占比 | NDCG@10 |
|---|---|---|---|
| 1–2 | 54,058 | 28.7% | **0.8944** |
| 3–4 | 43,661 | 23.2% | 0.9412 |
| 5–8 | 40,070 | 21.3% | 0.9492 |
| 9–10 | 11,043 | 5.9% | 0.9221 |
| 11–20 | 24,735 | 13.2% | 0.9482 |
| 21–50 | 12,410 | 6.6% | 0.9742 |
| 50+ | 2,110 | 1.1% | 0.9845 |

- **正样本多的用户已经接近满分**（0.97~0.98），那里没有空间。
- 缺口全在 npos ≤ 2 的 28.7% 用户身上：这一桶贡献了 `0.287 × 0.1056 ≈ 0.030` 的损失，**正好等于全部差距 0.0308**。
- 其中 npos=1 的 22,559 人 **hit@1 只有 0.7568**；npos=2 的 31,499 人 hit@1 是 0.8664。

### 5.3 结论与建议

要把整体从 0.9319 推到 0.9627，等于要求 npos≤2 的用户 hit@1 从 0.76/0.87 提到 **0.94 左右**——这是 leave-one-out 最难的场景，靠特征工程提升 20 个点不现实。**按当前候选集口径，+10% 大概率达不到，最终落点估计 0.94~0.95（+7%~+8%）。**

给成员三的方向（按性价比排序）：

1. **只优化 npos≤2 的用户**：这 28.7% 的用户决定了能不能达标。特征是「用户最近行为 + 该候选是否在其 genre 偏好内 + 与已看高分片的相似度」，别把精力放在已经 0.98 的活跃用户上。
2. **融合优先级**：`itemcf 分 + popular 分`（两路就够，LSH 别加）先做一版，看 npos≤2 桶有没有动。
3. **不要指望排序模型救全局**：LightGBM 在这类「1 正 + 100 负」的候选集上通常只涨 1~2 个点。

给成员四的写法（**建议现在就定，别等到 W6**）：

- 主表按 NDCG@10 报，同时报 **绝对提升（+5.67 点）** 和 **相对提升（+6.49%）**，两个都给。
- 加一行「**oracle 上限**」：NDCG@10 上限 1.0，Recall@10 上限 0.9110。用它解释为什么相对提升在接近天花板时天然收窄。
- 加一段「**候选集口径的影响**」：负样本 43% 是训练集里没出现过的冷门片（正样本电影训练正反馈次数中位数 4,126，负样本 1），所以热门排序天然占便宜，B0 被抬到 0.8752。**这是数据构造造成的，不是基线太强。**
- 如果老师认可，可以**补一套全库推荐口径**（不限制候选集）的对比：那种口径下 B0 会掉到 0.1 量级，相对提升远超 10%。这是唯一能"合规地"满足 +10% 字面要求的做法，但需要先跟老师确认口径。

---

## 6. LSH 是负结果，报告里必须这么写

题目要求的网格是 `bands ∈ {16,32,64} × rows ∈ {4,8,16}`。**网格内最好的配置（bands=64, rows=4）：**

| 指标 | 值 |
|---|---|
| NDCG@10 | **0.7127**（B0 是 0.8752，**连及格线都过不了**） |
| overlap@100（与精确 Top-100 邻居的重合率） | **0.1361** |
| 生成耗时 | 21.7 s（精确版 25.4 s，只有 1.2×） |
| 加上 MinHash 签名 63.2 s | 总耗时反而 **比精确版慢** |

交付文件用的是 `bands=64, rows=1`（NDCG 0.9133），**这是超出题目网格的诊断配置**，代价是：生成 118 s + 签名 63 s（比精确版慢 4.6 倍）、峰值内存 11.2 GB（精确版 3.8 GB）。**任何地方引用 0.9133 都必须同时说明这一点**，否则答辩会被追问"为什么不用 4/8/16"。

**为什么失败（报告的核心论点）**：精确 Top-100 邻居的 Jaccard 中位数只有 **0.083**（10 分位 0.007，90 分位 1.0）。banding 命中概率约 `s^rows`，`rows=4` 时对 0.083 只有 0.005%，`rows≥8` 基本为 0。LSH 能稳定捞到的只有 Jaccard≈1 的近似重复项，而真正的邻居全是低 Jaccard 对。**这是 MinHash 的适用边界，不是实现问题。**

报告里别写「LSH 没用」，写「**我们量化了 MinHash-LSH 在本数据上的适用边界：需要 rows≤2 才能保住 34%~48% 邻居召回，此时候选对涨到 700~3,900 万，加速完全消失**」。这是正面结论。

---

## 7. 复现

```powershell
cd D:\movie-rec
$env:PYTHONPATH=""

.\.venv\Scripts\python.exe -m pytest tests -q          # 15 个单测，约 10 s
.\.venv\Scripts\python.exe scripts\run_all.py --fast   # 评测器自检 + B0 + 表格，约 5 min
.\.venv\Scripts\python.exe scripts\run_all.py          # 全流程，约 70 min
.\.venv\Scripts\python.exe scripts\run_all.py --only itemcf tables   # 只跑某几步
```

依赖：pandas 3.1.0 / numpy 2.4.6 / scipy 1.17.1 / pyarrow 26.0.0 / pyspark 3.5.3 / **Java 17**（Spark 3.5 不支持 Java 21，`conf/local.toml` 里指 JDK 路径）。

数据不在仓库里（`data/` 已 gitignore），需要成员一的 `data.zip` 解压到 `data/official/`，然后跑 `scripts/verify_official_data.py` 验收。

---

## 8. 已知问题（别当成 bug）

1. **Spark 慢**：全量 val 441 s，单机分块版 142 s。固定开销（启动 + 读 CSV + pandas→Spark 转换）占 130~180 s，与核数无关，所以总加速比封顶 1.48×（4 核），16 核反而回落到 1.37×。**结论写成「本规模下 Spark 不划算，价值在数据量再涨一个数量级」**，这是诚实且有说服力的。
2. **Spark 打分阶段并行效率低**：每个 task 都要从广播变量重建 12.9 M 非零的矩阵，任务数一多固定开销线性增长。
3. **`%TEMP%` 问题**：本机进程令牌写不了系统临时目录，`movie_rec/spark.py` 的 `ensure_writable_temp()` 把 `TEMP` 指到 `data/_tmp`。队友机器正常的话可以不管。
4. **`results/neighbors_itemcf_k100.npz`（36 MB）** 是 LSH overlap 的基准邻居表，不提交 git，需要时 `scripts/run_itemcf.py` 重新生成。
5. **项目根目录的 `tmp*`、`pip-*` 目录**是历史调试残留，已 gitignore，不用管。

---

## 9. 有疑问找谁

| 问题 | 找谁 |
|---|---|
| 候选集/标签口径、test 阶段打分 | 成员一 |
| 分数怎么融合、排序模型怎么接 | 成员三 |
| 指标数字、报告章节、消融表 | 成员四 |
| 召回算法本身、Spark 作业、复现脚本 | 成员二（本仓库） |
