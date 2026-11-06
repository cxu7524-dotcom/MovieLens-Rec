# 成员二：相似度与 Spark 推荐

MovieLens 32M 分布式增量电影推荐系统 —— 成员二模块（评测器、热门基线、精确 ItemCF、MinHash LSH、Spark ItemCF）。

> **全队及格线 B0 = 热门基线**
> **NDCG@10 = 0.8752 / Recall@10 = 0.8639 / 覆盖率 = 0.1637**
> 任何召回模型在 val 上打不过这个数就不要提交。当前最好的单机模型是精确 ItemCF：**NDCG@10 = 0.9319（+0.0567）**。

---

## 0. 结果总览（val 阶段，20,281,548 个候选对，188,087 用户）

| 模型 | 配置 | NDCG@10 | Recall@10 | 覆盖率 | 说明 |
|---|---|---|---|---|---|
| random | 均匀随机 | 0.0876 | 0.0935 | 1.0000 | 评测器下限自检 |
| **popular (B0)** | 全局正反馈计数 | **0.8752** | **0.8639** | 0.1637 | 及格线 |
| popular-decay | 半衰期 30 天 | 0.8051 | 0.7973 | 0.1660 | 衰减反而更差 |
| popular-decay | 半衰期 90 天 | 0.8125 | 0.8099 | 0.1659 | |
| popular-decay | 半衰期 365 天 | 0.8205 | 0.8234 | 0.1655 | |
| **itemcf (精确)** | K=200，余弦 | **0.9319** | **0.8756** | 0.3312 | 交付 |
| LSH | bands=64, rows=1 | 0.9133 | 0.8612 | 0.2583 | 交付，但代价高 |
| itemcf (Spark) | topk=200, local[16] | 0.9319 | 0.8756 | 0.3312 | 与单机完全一致 |

**B0 为什么这么高？** 因为候选集的负样本基本是长尾随机项：正样本电影的训练正反馈次数中位数是 **4,126**，负样本只有 **1**（43% 的候选电影在训练正反馈里根本没出现过）。所以"按热度排"就能把绝大多数正样本顶到前 10。随机基线只有 0.0876，正好说明这个口径下的天花板和地板。

---

## 1. 运行环境

| 项 | 值 |
|---|---|
| Python | 3.11.5（`D:/movie-rec/.venv/Scripts/python.exe`） |
| pandas / numpy / scipy / pyarrow / pyspark | 3.1.0 / 2.4.6 / 1.17.1 / 26.0.0 / 3.5.3 |
| Java | 17.0.12（PyCharm JBR，`conf/local.toml` 指定；系统默认是 21，Spark 3.5 不支持） |
| 机器 | 16 核 / 17 GB，Windows 10 |
| 数据 | 官方交付 `data/official/`（train 25.5 M 行 / 正反馈 12.9 M / 46,847 部有正反馈的电影） |

安装 scipy 与 pytest（首次）：

```powershell
cd D:\movie-rec
$env:PYTHONPATH=""
.\.venv\Scripts\python.exe -m pip install -i https://repo.huaweicloud.com/repository/pypi/simple "scipy>=1.14" pytest
```

---

## 2. 一条命令重跑

```powershell
cd D:\movie-rec
$env:PYTHONPATH=""

# 全流程（约 70 分钟：单测 → 评测器自检 → B0 → ItemCF → LSH → Spark → 扩展性 → 表格）
.\.venv\Scripts\python.exe scripts\run_all.py

# 只跑不重的部分（约 5 分钟，用于改完代码先验一遍）
.\.venv\Scripts\python.exe scripts\run_all.py --fast

# 单独跑某一步
.\.venv\Scripts\python.exe scripts\run_all.py --only itemcf tables
```

`run_all.py` 只是按顺序调用下面这些脚本，没有第二套逻辑；手工单跑也可以：

```powershell
# 0) 单测：评测器玩具例子 + recall 模块（余弦 TopK / LSH 分桶）
.\.venv\Scripts\python.exe -m pytest tests -q

# 1) 评测器自检：随机分数 NDCG 必须 < 0.1
.\.venv\Scripts\python.exe scripts\check_evaluator.py

# 2) 热门基线 B0（约 80 秒）
.\.venv\Scripts\python.exe scripts\run_popular.py

# 3) 精确 ItemCF（约 2.5 分钟，峰值 3.8 GB）
.\.venv\Scripts\python.exe scripts\run_itemcf.py

# 4) LSH 参数扫描 + 交付分数（约 12 分钟，峰值 11 GB）
.\.venv\Scripts\python.exe scripts\run_lsh.py

# 5) Spark ItemCF（约 8 分钟）
.\.venv\Scripts\python.exe scripts\run_spark_itemcf.py --stage val --topk 200 --master "local[16]"

# 6) 三张交付表 + 扩展性曲线（只读 results/*.json，秒级）
.\.venv\Scripts\python.exe scripts\report_tables.py
.\.venv\Scripts\python.exe scripts\plot_scaling.py
```

单独评测任意分数文件：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --scores results\scores_itemcf_val.csv --stage val --name itemcf_val
```

---

## 3. 第 1 步：本地评测器（`src/movie_rec/eval.py` + `scripts/evaluate.py`）

**口径**（写死在 `conf/base.toml`，全队一致）：

- 正样本：`validation.csv` 里 `rating >= 4.0`；test 阶段没有标签，只校验形状与覆盖率。
- NDCG@10：gain = 1，discount = `1/log2(p+1)`，IDCG 用该用户在候选集内的正样本数算。
- Recall@10：命中数 / 该用户在候选集内的正样本数。
- 覆盖率：所有人 top-10 出现过的电影数 / 候选集里出现过的电影数（87,585）。
- 缺失对填 0（交付契约），候选集外的分数忽略并计数。

**实现要点**：一次 `np.lexsort` 把 20 M 行按 `(userId, 分数降序)` 排好，然后按用户块取前 k 列；每用户一个 10 次的向量化循环，没有 Python 级用户循环。同分按候选文件顺序决定，可复现。

**单测**：`tests/test_eval.py` 用手工算的 3 用户玩具例子（含并列分数、k 大于候选数两种情况）逐项对比，误差 < 1e-9；另有一组 120 用户随机数据与朴素实现对照。

**验收**：

```text
随机分数：NDCG@10=0.0876  Recall@10=0.0935  coverage=1.0000  users=188,087
20,281,548 对 15.65 s（0.083 ms/用户）
EVAL OK: random baseline NDCG@10=0.0876 < 0.1
```

---

## 4. 第 2 步：热门基线 B0（`src/movie_rec/recall/popular.py`）

两种口径，都只用训练正反馈：

- **count**：`score(i) = 训练正反馈次数`。全局 Top-10 是肖申克、低俗小说、黑客帝国……（见 `results/baseline_popular.json`）。
- **decay**：`score(u,i) = Σ_k exp(-λ·(c_u - t_k))`，`λ = ln2 / 半衰期`，`c_u` 是用户 cutoff。

实现上按因子分解避免 20 M × 事件数的循环：

```text
score(u,i) = exp(-λ(c_u - T)) · A_i,   A_i = Σ_k exp(-λ(T - t_k)),  T = 最新训练时间戳
```

**一个重要结论**：`exp(-λ c_u)` 对同一用户是常数，所以**时间衰减热门的排序与全局衰减热度完全等价**——它并不带来任何个性化。三个半衰期都低于 count 版本（0.805 / 0.813 / 0.821 vs 0.875），因为衰减把老经典打压下去，而验证集的正样本恰恰多是老经典。

交付分数按用户内最大值归一化到 [0,1]（单调变换，NDCG 不变，但方便成员三做加权融合）。

**B0 = count 版本：NDCG@10 = 0.8752 / Recall@10 = 0.8639 / 覆盖率 0.1637**（打分 1.9 s，评测 2.9 s）。

---

## 5. 第 3 步：精确 item-item 相似度（`src/movie_rec/recall/itemcf.py`）

- 训练正反馈 → `scipy.sparse` 二值矩阵 `X`：200,571 用户 × 46,847 电影，12,905,652 个非零，密度 0.137%。
- 余弦相似度，**按行分块**：每块 2,000 部电影，块内算 `X[:, block].T @ X` 得 `2000 × 46,847` 稠密片（float32，375 MB），归一化后取 TopK，然后丢弃整块。全量稠密矩阵需要 8.8 GB，这里从不物化。
- 打分 `score(u,i) = Σ_{j∈I_u} sim(i,j)`，只对候选集里的 (u,i) 取值；按用户切片取该用户候选电影的 K 邻居行，与用户历史向量相乘。

| 指标 | 值 |
|---|---|
| 邻居边数 | 8,874,088（K=200） |
| 相似度构建 | **25.4 s**，24 个块 |
| 峰值内存 | **3,764 MB**（< 8 GB 要求） |
| 全量打分 | 52.7 s（385,050 对/秒） |
| K 扫描（20k 用户抽样） | K=50 → 0.9216，K=100 → 0.9292，**K=200 → 0.9315** |

**验收：NDCG@10 = 0.9319（B0 0.8752，+0.0567），覆盖率 0.3312，峰值内存 3.76 GB。**

`tests/test_recall.py` 用 4 用户 6 电影的玩具矩阵对照稠密余弦：TopK 取值与稠密结果一致、对角线自相似被清零、`block_items` 取 1/2/3/100 时邻居数都不超 K 也不丢邻居、候选打分等于手算的点积、未出现过的电影得 0。

精确邻居表存到 `results/neighbors_itemcf_k100.npz`，作为 LSH 的 overlap@100 基准。

---

## 6. 第 4 步：LSH 近似召回（`src/movie_rec/recall/lsh.py`）

MinHash（multiply-shift 哈希，1024 个，签名 192 MB，构建 63 s）+ banding。只吃二值矩阵，不吃评分。所有 (bands, rows) 组合复用同一份签名的前缀，所以 13 组配置只建一次签名。

| bands | rows | 候选对 | overlap@100 | 生成 s | 峰值 MB | NDCG@10（抽样） |
|---|---|---|---|---|---|---|
| 16 | 4 | 1,949,611 | 0.0977 | 4.7 | 3,402 | 0.6383 |
| 16 | 8 | 1,162,847 | 0.0589 | 4.0 | 3,402 | 0.5813 |
| 16 | 16 | 1,093,766 | 0.0547 | 4.3 | 3,402 | 0.5767 |
| 32 | 4 | 2,297,777 | 0.1162 | 8.8 | 4,338 | 0.6736 |
| 32 | 8 | 1,214,187 | 0.0619 | 7.2 | 4,338 | 0.5840 |
| 32 | 16 | 1,093,908 | 0.0547 | 9.6 | 4,338 | 0.5767 |
| 64 | 4 | 2,742,843 | 0.1361 | 21.7 | 6,136 | 0.7127 |
| 64 | 8 | 1,294,517 | 0.0669 | 14.5 | 6,136 | 0.5869 |
| 64 | 16 | 1,094,092 | 0.0548 | 15.5 | 6,136 | 0.5780 |
| *16* | *1* | 20,595,356 | 0.3624 | 31.6 | 6,469 | 0.8876 |
| *32* | *1* | 27,410,128 | 0.4227 | 51.7 | 7,726 | 0.9025 |
| *64* | *1* | 38,725,787 | **0.4790** | 118.1 | 10,477 | **0.9141** |
| *64* | *2* | 6,979,207 | 0.3352 | 33.0 | 10,477 | 0.9036 |

斜体行是超出题目要求网格的诊断配置（`rows=1` 时 banding 退化成"任一哈希相同即同桶"）。

**结论（验收要求的那句话）**：

> **LSH 在 required grid 里最好的配置（bands=64, rows=4）只保住 13.6% 的精确 Top-100 邻居，换来的是 21.7 s 生成时间——比精确算法的 25.4 s 快 1.2 倍，NDCG 从 0.9319 掉到 0.7127。**
> 即使放开到 rows=1（overlap@100 = 0.479，NDCG 0.9133），也要 118 s 生成 + 63 s 签名，比精确版慢 4.6 倍，还要多占 6.7 GB 内存。
> **即：本数据上 LSH 是"损失 52%–86% 的邻居召回，换取 0.2×–1.2× 的速度"，不划算。**

**为什么**：精确 Top-100 邻居的 Jaccard 中位数只有 **0.083**（10 分位 0.007，90 分位 1.0）。banding 撞上概率约 `s^rows`，`rows=4` 时对 0.083 的 Jaccard 只有 0.005% 的概率，`rows≥8` 基本为 0。LSH 能稳定捞到的只有 Jaccard≈1 的近似重复项（同一小圈子用户看过的冷门片），而真正的邻居全是低 Jaccard 对——这是 MinHash 的适用边界，不是实现问题。要召回它们只能把 `rows` 压到 1~2，那时候候选对涨到 2000–4000 万，加速就消失了。

交付文件 `results/scores_lsh_val.csv` 用 bands=64/rows=1（NDCG 最高）：**NDCG@10 = 0.9133 / Recall@10 = 0.8612 / 覆盖率 0.2583**，打分 69.2 s，进程峰值 **11.2 GB**（比精确版的 3.8 GB 高 3 倍）。

---

## 7. 第 5 步：Spark ItemCF（`src/movie_rec/spark_jobs/itemcf.py`）

**为什么不写教科书式的 self-join**：`ratings ⋈ ratings on userId` 会物化全部共现对，本数据是 `Σ_u |I_u|² = 25.6 亿` 对（用户历史中位数 34、p99 444、最大 4,420），16 GB 笔记本上光 shuffle 就要几十 GB。同样的共现矩阵可以不算出任何一对就得到：

```text
C[block, :] = X[:, block].T @ X      # 稀疏×稀疏，每个用户的贡献在内核里累加
```

浮点运算量一样，但没有 25.6 亿条中间记录。这是本机唯一跑得通的分布式写法，README 之前"只取正反馈最多的 5,000 部电影"的退路也就不需要了。

**分布式的部分**：

1. **共现块**：47 个 item 块（每块 1,000 部）交给 47 个 task 并行，每个 task 从广播变量（12.9 M 非零，103 MB）重建矩阵，算完自己的 `1000 × 46,847` 片、取 TopK、丢掉。
2. **广播打分**：候选对按 `userId` 哈希重分区 + 分区内排序，保证一个用户不跨分区、一个 task 只缓存一个用户的 ~108 行；邻居表广播到每个 task。

| 指标 | 值 |
|---|---|
| 训练数据读取（Spark CSV → 矩阵） | 45.9 s |
| 共现 + 余弦 + TopK | 51.0 s，47 块，**8,874,088 条边**（与单机完全一致） |
| pandas → Spark DataFrame 转换（20.28 M 行） | 227.4 s |
| 候选打分（20.28 M 对） | 103.1 s |
| 总计 / 峰值内存 | 441.0 s / 5,128 MB |
| **NDCG@10** | **0.9319，与单机精确版差异 0.0000**（要求 < 0.001） |

test 阶段同样跑通，产出 `results/scores_spark_itemcf_test.csv`（20,660,431 对，打分 101.7 s）。

扩展性（并行度 1/2/4/8/16，固定 3 万用户抽样 = 3,236,361 对，表见 `results/table_scaling.md`，曲线见 `results/scaling_curve.svg`）：

| 并行度 | 总耗时 s | 共现 s | 打分 s | 转换 s | 吞吐 pairs/s | 加速比 | 并行效率 |
|---|---|---|---|---|---|---|---|
| 1 | 276.5 | 121.8 | 42.6 | 33.3 | 11,705 | 1.00 | 1.00 |
| 2 | 220.0 | 71.4 | 29.6 | 42.7 | 14,711 | 1.26 | 0.63 |
| 4 | 187.2 | 50.2 | 23.6 | 43.6 | 17,288 | 1.48 | 0.37 |
| 8 | 188.7 | 50.3 | 29.3 | 45.4 | 17,151 | 1.47 | 0.18 |
| 16 | 201.9 | 48.3 | 50.2 | 44.9 | 16,030 | 1.37 | 0.09 |

怎么读这张表：

- **只有共现阶段真正并行**：121.8 s → 48.3 s（16 核 2.5 倍）。它是唯一随核数下降的项。
- **打分阶段 4 核之后反弹**（23.6 → 29.3 → 50.2 s）：分区数跟着 `defaultParallelism×2` 涨，而每个 task 都要从广播变量重建一次 12.9 M 非零的矩阵，固定开销随任务数线性增加。
- **固定开销吃掉加速比**：Spark 启动 + 读 CSV（45–70 s）+ pandas→Spark 转换（~44 s）+ 评测，合计 130–180 s，与核数无关。所以总加速比封顶在 **1.48 倍**（4 核），16 核反而回落到 1.37 倍。

> **结论：这个规模上 Spark 不划算——单机分块版 142 s 跑完全部 20.28 M 对，Spark 441 s。Spark 的价值在数据量再涨一个数量级之后：本实现的共现阶段已经与候选集规模无关（只依赖交互矩阵），候选打分才是需要横向扩展的部分。**

---

## 8. 交付物

### 给成员三（打分文件，`userId,movieId,score` 三列，覆盖候选集每一对）

| 文件 | 行数 | 大小 |
|---|---|---|
| `results/scores_itemcf_val.csv` | 20,281,548 | 467 MB |
| `results/scores_itemcf_test.csv` | 20,660,431 | 476 MB |
| `results/scores_lsh_val.csv` | 20,281,548 | 467 MB |
| `results/scores_lsh_test.csv` | 20,660,431 | 476 MB |
| `results/scores_popular_val.csv` | 20,281,548 | 467 MB |
| `results/scores_spark_itemcf_test.csv` | 20,660,431 | — |

分数范围：itemcf 是余弦累加（0 ~ 数十），popular 与 lsh 归一化到 [0,1]。**同一文件内部可比，跨文件不可比**，融合前请各自归一化。

### 给成员四

| 文件 | 内容 |
|---|---|
| `results/table_quality.md` / `.csv` | 近似质量表：NDCG@10 / Recall@10 / 覆盖率 / overlap@100 |
| `results/table_efficiency.md` / `.csv` | 效率对比：耗时、内存、单用户延迟、吞吐 |
| `results/table_scaling.md` / `.csv` | 扩展性：并行度 1/2/4/8/16 的耗时、加速比、并行效率 |
| `results/scaling_curve.svg` | 扩展性曲线（左：各阶段耗时；右：加速比，含 1.0× 参考线） |
| `results/lsh_val.json` | LSH 全部 13 组参数 + overlap + 交付结果 |
| `results/itemcf_val.json` | K 扫描、相似度构建、峰值内存 |
| `results/spark_scaling_val.json` | 扩展性原始数据 |
| `results/manifest_*.json` | 数据指纹（size/mtime/sha1 前 4 MiB）、参数、种子、环境 |

### 给全队

`src/movie_rec/{eval,dataset,runlog}.py`、`src/movie_rec/recall/`、`src/movie_rec/spark_jobs/`、`scripts/`（含 `run_all.py` 一键重跑）、`tests/`（15 个单测，`pytest tests -q`）。

---

## 9. 数据契约（成员一交付，未改动）

CSV 路径在 `conf/base.toml` 的 `[data.official]`。已验收事实：train 25,520,897 行 / 200,948 用户 / 正反馈 12,905,652；val 候选 20,281,548 对 / 188,087 用户；test 候选 20,660,431 对 / 191,007 用户；验证阶段 100% 正样本都在候选集内。

本模块新增的读取层 `src/movie_rec/dataset.py` 把 CSV 一次性转成 `data/_cache/*.parquet`（按 size+mtime 做 key），原始文件只读不写。所有脚本共用 `load_settings()`，不另起配置。

---

## 10. 已知问题与下一步

1. **LSH 在本数据上不划算**，见第 6 节。如果课程要求必须给出正收益，需要换相似度口径（例如对 item 做 TF-IDF 加权或按用户活跃度归一化），把邻居的 Jaccard 抬到 0.3 以上——那是另一套实验。
2. **Spark 打分阶段是瓶颈**（103 s 计算 + 227 s pandas→Spark 转换 / 20.28 M 对）。优化方向：候选直接由 Spark 从 CSV 读入，省掉 227 s 的 DataFrame 转换；把邻居表按 item 哈希分桶后与候选做 join，避免逐用户 Python 循环。
3. **test 阶段没有标签**，`scores_*_test.csv` 只能交给成员一验；覆盖率 0.3285（itemcf）与 val 的 0.3312 基本一致，说明两阶段分布没有漂移。
4. **`%TEMP%` 问题**：本机进程令牌不能写系统临时目录，Spark 启动脚本会报 `拒绝访问 / 找不到 spark-class-launcher-output`。`movie_rec/spark.py` 的 `ensure_writable_temp()` 把 `TEMP`/`TMP` 指到 `data/_tmp` 后正常；队友机器上若正常可以不管。
5. 历史残留：项目根目录的 `tmp*` 目录和 `data/_probe*` 是早前调试产物，未清理。
