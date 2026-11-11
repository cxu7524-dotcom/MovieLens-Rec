# 成员二：相似度 / LSH / Spark 分布式推荐（v2：评分预测）

课程项目「MovieLens 分布式增量电影推荐系统」。**2026-11 起正式任务改为评分预测**：对 `test_pairs.csv` 的 100,000 个 (userId, movieId) 预测评分，用 **RMSE** 评价。

> ## 全队及格线 B0 = 全局均值预测
> **RMSE = 1.047271**（val，99,998 对）
> 任何模型在 val 上打不过这个数就不要提交。当前最好的单机模型是 ItemCF（K=200, β=20）：**RMSE = 0.926035**。
> 任务书「相对基线提升 ≥10%」按 `RMSE ≤ B0 × 0.9 = 0.942544` 理解（**口径待老师确认**，见第 10 节）。

> ⚠️ **本仓库有两套协议，别混用**
> - **v2（本文件主线，2026-11）**：RMSE 评分预测，数据 `data/hybrid/`，产出 `results/pred_test.csv`。
> - **v1（已作废，2026-11-05）**：NDCG@10 / Top-10 候选排序，数据 `data/official/`。**所有 v1 数字都不能用于本次验收**，代码保留在 `recall/popular.py`、`scripts/run_itemcf.py` 等文件里，只作报告里的「推荐模块展示」补充实验，见第 11 节。

---

## 0. 结论速览（val，99,998 对 / 5,498 用户，pair 等权）

| 模型 | RMSE | MAE | 相对 B0 | 说明 |
|---|---|---|---|---|
| random（自检下限） | 1.842889 | 1.505075 | -76.0% | 评测器下限 |
| **global mean = B0** | **1.047271** | 0.828663 | 0 | 及格线 |
| user mean | 1.111734 | 0.868033 | **-6.16%** | 比全局均值还差 |
| item mean | 0.948220 | 0.740391 | +9.46% | 物品偏差是主力 |
| **bias μ+b_u+b_i（λ=20）** | **0.929663** | 0.722442 | **+11.23%** | 已过 10% 目标 |
| bias + 时间漂移 | 0.931461 | 0.723627 | +11.06% | 时间项无增益 |
| itemcf K=200, β=20 | 0.926035 | 0.718656 | **+11.58%** | 单机最优，交付 |
| lsh 近似邻居 | 0.929677 | 0.722439 | +11.22% | 退化为 bias 水平 |
| spark ALS（rank=20, iter=20, reg=0.1） | 0.987469 | 0.775500 | +5.71% | 比 B0 好，但不如 bias / ItemCF |

三个直接结论：

1. **物品偏差贡献了几乎全部收益**（user mean 反而比全局均值差 6.16%：验证用户是重活跃用户，个人均值被长尾低分拉偏）。
2. **ItemCF 只比 bias 好 0.0036（0.39%）**。余弦相似度用二值共现，和评分残差的相关性弱；K 从 50 加到 200 也只多 0.0004。
3. **LSH 在本数据上不划算**：overlap@100 最高 0.91%，RMSE 退回 bias 水平，算上签名构建是 0.68× 速度（更慢）。如实报负结果，见 7.2。

---

## 1. 环境与数据

| 项 | 值 |
|---|---|
| Python | 3.11.5（`D:/movie-rec/.venv/Scripts/python.exe`） |
| 依赖 | pandas 3.1.0 / numpy 2.4.6 / pyarrow 26.0.0 / pyspark 3.5.3 / scipy 1.17.1 |
| Java | 17.0.12（PyCharm JBR，`conf/local.toml`；系统默认 21，Spark 3.5 不支持） |
| 机器 | 16 核 / 17 GB，Windows 10 |
| 数据集 | `hybrid-half-half-latest100k-v1`（seed 20261106），从 `D:/qq/文件/训练数据集/` 原样复制到 `data/hybrid/`，原始目录只读未动 |

数据事实（`scripts/verify_hybrid_data.py` 实测，全绿）：

- `ratings.csv` 31,900,204 行 / 396,060 用户 / 142,609 部有评分的电影；rating 均值 **3.540359**、std **1.059044**；正反馈(≥4) 15.89 M
- `test_pairs.csv` **100,000 行 3 列**（无 rating）；5,498 用户，人均 18.2 条、中位 4、最多 1,048
- test 与训练 **零重叠**（0/100,000），test 用户/电影训练覆盖率 **100%**（无冷启动）
- MovieLens 半区 49,953 对 / 合成半区 50,047 对；两半区评分均值差 0.000481
- `tags.csv` 1,991,034 条（26,874 用户 / 88,595 电影）—— 本次未使用，留给后续特征
- `movies.csv` 175,170 部，其中 32,561 部没有任何训练评分

> **术语**：142,609 是「训练评分中出现过的电影数」，不是全库；全库是 175,170。

---

## 2. 一条命令重跑

```powershell
cd D:\movie-rec
$env:PYTHONPATH=""

# v2 全流程（约 2 小时：验收 → val → 评测器 → 基线 → ItemCF → LSH → ALS → test → 表格）
.\.venv\Scripts\python.exe scripts\run_all.py

# 跳过重活（约 1 分钟，改完代码先验一遍）
.\.venv\Scripts\python.exe scripts\run_all.py --fast

# 单步
.\.venv\Scripts\python.exe scripts\run_all.py --only itemcf-rating report-v2

# 旧 v1 NDCG 展示链路
.\.venv\Scripts\python.exe scripts\run_all.py --legacy
```

`run_all.py` 只是按顺序调用下面这些脚本，没有第二套逻辑。手工单跑：

```powershell
# Step 1 数据验收（18.8 s）
.\.venv\Scripts\python.exe scripts\verify_hybrid_data.py
# Step 2 val 切分（24.5 s）
.\.venv\Scripts\python.exe scripts\make_val_split.py
# Step 3 评测器自检 + 单测
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\check_evaluator.py
# Step 4 基线阶梯（103.5 s）
.\.venv\Scripts\python.exe scripts\run_baselines.py
# Step 5a ItemCF（265.8 s，峰值 3.9 GB）
.\.venv\Scripts\python.exe scripts\run_itemcf_rating.py
# Step 5b LSH（888 s，峰值 12.6 GB）
.\.venv\Scripts\python.exe scripts\run_lsh_rating.py
# Step 6 ALS（调参 + 全量）与扩展性
.\.venv\Scripts\python.exe scripts\run_als.py
.\.venv\Scripts\python.exe scripts\run_als.py --scaling
# Step 7 test 预测 + 表格 + 曲线
.\.venv\Scripts\python.exe scripts\predict_test.py --model auto
.\.venv\Scripts\python.exe scripts\report_v2.py
.\.venv\Scripts\python.exe scripts\plot_scaling.py --als
```

单独评测任意预测文件：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\pred_test.csv --stage test --name pred_test
```

---

## 3. Step 1：数据验收 + 画像（`scripts/verify_hybrid_data.py`）

六组检查，任一失败即非零退出，可作为后续步骤的闸门。输出 `data/hybrid/hybrid_data_check.json`。

1. 八个文件齐备、表头与契约一致
2. `ratings.csv` 行数/用户数/电影数/评分范围与 `summary.json` 一致，无重复 (userId, movieId)
3. 两半区行数（15,950,149 / 15,950,055）与 summary 一致，评分分布接近
4. `test_pairs.csv` 100,000 行、无 rating 列、与训练零重叠、无冷启动、无重复
5. 每个 test pair 都晚于该用户的历史（488 条与最后一条训练记录同时间戳，属同批次，不构成未来信息）
6. tags 规模与 movies 覆盖

**验收输出**：`HYBRID DATA OK / scoring job: 100,000 test pairs, 5,498 users, 0 cold start`。

> 时间戳陷阱：`summary.json` 的 `hidden_cutoff_timestamp=1693375424` 是**全局**截断，训练集里有 31,806 条晚于它的记录，属于其他用户的合法数据，**不能按它过滤**。脚本把这条写进了报告。

---

## 4. Step 2：本地 val 切分（`scripts/make_val_split.py`）

官方隐藏集没有标签，必须自造 val，并复刻 `summary.json.hidden_selection_constraints`：

- **按用户时序**：隐藏每个 donor 用户**连续的最后 h 条**评分（不是随机抽样，否则会留下"未来"记录）
- `minimum_earlier_training_ratings_per_test_user = 5`：只接受 `count ≥ h + 5` 的 donor
- `minimum_training_ratings_per_test_movie = 1`：尾部电影必须还有别的评分，杜绝冷启动
- **每个 donor 隐藏的条数 1:1 复制官方 test 的每用户条数**，人均 18.2 / 中位 4 / 最多 1,048 与原分布一致
- donor 与 5,498 个 test 用户**不相交**，按原始评分条数配对（匹配重活跃用户分布）

产出 `data/hybrid/val_train.parquet`（31,800,204 行）、`val_pairs.parquet`（99,998 对，带 rating）、`val_split_report.json`。

**验收数字**：

```text
val pairs 99,998 | users 5,498 | mean 18.2 median 4 max 1048
global-mean baseline on val: RMSE=1.047271 MAE=0.828663 (mu=3.540358)
strictly earlier than history: 0 | ties: 1242
VAL SPLIT OK
```

### ⚠️ 关于「全局均值 RMSE 应落在 1.05~1.08」

实测 **1.047271**，比提示词给的带宽下限低 0.0027（0.26%）。这不是切错，报告里同时记录了两个对照：

| 口径 | RMSE | 说明 |
|---|---|---|
| 本 val（用户时序尾部） | **1.047271** | 时序尾部天然方差更小 |
| 同规模随机 holdout | **1.057734** | 等于训练集 std 1.059044，落在带内 |

两者差 0.0105 全部来自「时序尾部」这个构造：越靠后的评分越接近用户的稳定偏好，方差更小。**若强行把 val 调进 1.05~1.08，只能改成随机抽样，那就不再复刻官方的 per-user temporal order**。此处选择忠于官方切法，并把偏差写进报告。

---

## 5. Step 3：RMSE 评测器 v2（`src/movie_rec/eval.py` + `scripts/evaluate.py`）

- 输入 `userId,movieId,pred_rating`，**必须覆盖该阶段每一对、且不多不少**
- 硬校验：重复对、缺对、多对、NaN/Inf、越界（不在 [0.5, 5.0]）→ 直接抛错
- 指标：`rmse / mae / n_pairs / n_users / elapsed_s`，默认 pair 等权（micro），可切 `weighting="user"`（macro：先算每用户 MSE 再开方）
- 同时输出两半区（按 `id_ranges.json` 的电影 ID 分界）的 RMSE/MAE

**单测**：`tests/test_eval_rmse.py` 用手算玩具例子逐项比对（误差 < 1e-9），覆盖并列、缺对、重复、越界、NaN；`tests/test_baselines.py` 覆盖偏置模型的闭式解、未知 id 回退、clip 与两种平均方式。合计 **36 个单测全绿**（1.0 s）。

**验收**：

```text
mean prediction  : RMSE=1.047271 MAE=0.828663 bias=-0.000332
random prediction: RMSE=1.848945 MAE=1.511707
user-weighted (mean pred): RMSE=1.067203
split report recorded 1.047271, recomputed 1.047271
EVAL OK: mean prediction reproduces the hand value, random is 1.77x worse
```

---

## 6. Step 4：评分预测基线阶梯（`src/movie_rec/baselines.py`）

| 层级 | RMSE | MAE | 相对 B0 | 参数 |
|---|---|---|---|---|
| **global (B0)** | **1.047271** | 0.828663 | +0.00% | μ=3.540358 |
| user mean | 1.111734 | 0.868033 | **-6.16%** | λ=0 |
| item mean | 0.948220 | 0.740391 | +9.46% | λ=0 |
| **bias (λ=20)** | **0.929663** | 0.722442 | **+11.23%** | 15 轮交替最小二乘 |
| bias + 时间漂移 | 0.931461 | 0.723627 | +11.06% | λ_drift=10 |

λ 扫描：1 → 0.972066，5 → 0.939643，10 → 0.931596，**20 → 0.929663**，50 → 0.932217。

要点：

- **user mean 比全局均值更差**（-6.16%）。测试用户是重活跃用户，他们的个人均值被大量长尾低分拉低，直接当预测值反而偏。这一条对「要不要做用户归一化」是个警告。
- **item mean 一个人就拿到 9.46%**，说明物品偏差是主要信号。
- 偏置模型在 λ=20 达到 0.929663，**已经越过 10% 目标线 0.942544**。
- 时间漂移项（每用户评分随时间的线性趋势）没有增益（+0.0018 变差），不加。

正式 B0 = **1.047271**，已写进本文件顶部与 `results/baselines_val.json`。

---

## 7. Step 5：ItemCF 评分预测 + LSH 近邻加速

### 7.1 ItemCF（`src/movie_rec/recall/itemcf.py` + `scripts/run_itemcf_rating.py`）

预测公式（在 bias 上做残差加权）：

```text
r̂(u,i) = b_ui + Σ_{j∈N_K(i)∩I_u} sim(i,j)·(r_uj − b_uj) / Σ|sim(i,j)| × cnt/(cnt+β)
```

- 相似度只用 `val_train` 计算；142,609 部电影，**分块 1000**（2000×142,609×4B ≈ 1.1 GB/块，16 GB 内存要留余量）
- 余弦用二值共现（看过/没看过），不是评分相关
- 收缩项 `cnt/(cnt+β)`：K=200 时一个用户的 200 部历史里只有 1~2 部是候选的邻居，不收缩的话修正项基本是噪声

| 指标 | 值 |
|---|---|
| 邻居边数 | 27,931,725（K=200） |
| 相似度构建 | 177.5 s，143 块 |
| 峰值内存 | **3,880 MB** |
| 总耗时 | 265.8 s |
| **val RMSE** | **0.926035**（bias 0.929663，**+0.003628 / +0.39%**） |
| 两半区 | ML 0.885868 / 合成 0.961735 |

K/β 扫描（完整 15 组见 `results/v2_table_params.md`）：K=50 β=10 → 0.926447；K=100 β=20 → 0.926124；**K=200 β=20 → 0.926035**。**K 从 50 加到 200 只多 0.0004**，说明邻居数量不是瓶颈，相似度本身太弱。收缩项才是关键：β=0 时 RMSE 是 0.9965~1.0266，比 bias 还差。

### 7.2 LSH 近邻加速（`src/movie_rec/recall/lsh.py` + `scripts/run_lsh_rating.py`）

用途改造：不再做全库候选召回，而是**给定 val 里出现的 17,024 部电影，快速找它们的 Top-100 相似邻居**，再看这个近似够不够用。

MinHash（1024 哈希，签名 584 MB，构建 139.9 s）+ banding，扫 9 组参数：

| bands | rows | 候选对 | overlap@100 | 生成 s | 峰值 MB | RMSE |
|---|---|---|---|---|---|---|
| 16 | 4 | 8,595,684 | 0.0038 | 19.8 | 6,838 | 0.929657 |
| 16 | 8 | 4,281,297 | 0.0012 | 17.2 | 6,838 | 0.929665 |
| 16 | 16 | 4,007,485 | 0.0011 | 17.8 | 6,838 | 0.929665 |
| 32 | 4 | 16,566,515 | 0.0062 | 54.7 | 9,027 | 0.929656 |
| 32 | 8 | 4,836,913 | 0.0013 | 29.5 | 9,027 | 0.929665 |
| 32 | 16 | 4,008,304 | 0.0011 | 30.6 | 9,027 | 0.929665 |
| **64** | **4** | 20,284,117 | **0.0091** | 100.0 | 12,589 | 0.929677 |
| 64 | 8 | 5,424,241 | 0.0016 | 56.5 | 12,589 | 0.929663 |
| 64 | 16 | 4,010,297 | 0.0011 | 68.4 | 12,589 | 0.929665 |

**结论（验收要求的那句话）**：

> 精确近邻：构建 163.6 s，RMSE **0.926127**。
> LSH 最好配置（bands=64, rows=4）：**overlap@100 只有 0.91%**，RMSE 0.929677（**+0.003550**，即退回 bias 水平），生成 100.0 s（**1.64×** 精确构建），**加上 139.9 s 签名构建后只有 0.68×（更慢）**，峰值内存 12.6 GB 是精确版的 2.7 倍。
> **即：LSH 用「损失 99% 的邻居召回 + RMSE 退化到 bias + 更慢 + 3 倍内存」换不到任何东西，本数据上完全不划算。**

原因和 v1 一致且更极端：本数据 item 的 Jaccard 中位数极低（142,609 部电影，中位评分条数只有 4），banding 的撞桶概率约 `s^rows`，rows≥4 时对低 Jaccard 对几乎为 0。能撞上的只有「同一小撮用户看过的冷门片」，而它们不是余弦 Top-100。

---

## 8. Step 6：Spark ALS 显式评分预测 + 扩展性

`src/movie_rec/spark_jobs/als.py` + `scripts/run_als.py`。

- `ALS(implicitPrefs=False)`，显式评分
- `coldStartStrategy="nan"`：未知 user/item 不丢行，用第 4 步 bias 预测**逐对**兜底（调参时 10% 样本有大量未出现的电影，用全局均值兜底会把 RMSE 抬到 1.5+，配置之间就没有区分度了）
- 预测 clip 到 [0.5, 5.0]；训练数据走 `spark.read.parquet`，不做 pandas 往返（v1 那次 20 M 行 `createDataFrame` 花了 227 s）

### 8.1 12 组调参（10% 样本 = 3,180,386 条）

| rank | iterations | regParam | RMSE | MAE | fit s |
|---|---|---|---|---|---|
| 20 | 10 | 0.01 | 1.587508 | 1.221373 | 36.8 |
| 20 | 10 | 0.1 | 1.176377 | 0.922823 | 31.1 |
| 20 | 20 | 0.01 | 1.624415 | 1.252698 | 43.2 |
| **20** | **20** | **0.1** | **1.135242** | 0.889690 | 46.8 |
| 50 | 10 | 0.01 | 1.784025 | 1.390601 | 72.2 |
| 50 | 10 | 0.1 | 1.189260 | 0.935920 | 60.1 |
| 50 | 20 | 0.01 | 1.579932 | 1.230061 | 103.8 |
| 50 | 20 | 0.1 | 1.149963 | 0.903981 | 96.8 |
| 100 | 10 | 0.01 | 1.714678 | 1.339749 | 169.5 |
| 100 | 10 | 0.1 | 1.198175 | 0.944035 | 167.4 |
| 100 | 20 | 0.01 | 1.402365 | 1.101151 | 316.6 |
| 100 | 20 | 0.1 | 1.155098 | 0.909484 | 338.5 |

三条读数：

- **regParam 是唯一有决定性的参数**：0.01 全线过拟合（1.40~1.78），0.1 全部好 0.4~0.6。
- **rank 从 20 加到 100 没有改善**（1.135242 → 1.155098，反而略差），而拟合耗时涨了 7 倍。10% 样本喂不饱高 rank。
- 这些 RMSE 含冷启动兜底，**只用于给配置排序，不是模型最终 RMSE**。

### 8.2 全量拟合（最优配置 rank=20 / iter=20 / reg=0.1）

| 指标 | 值 |
|---|---|
| 训练评分 | 31,800,204 |
| 训练耗时 | **113.7 s** |
| 吞吐 | 279,768 条/秒 |
| **val RMSE** | **0.987469** |
| MAE | 0.775500 |
| 两半区 | MovieLens 0.970328 / 合成 1.003081 |

**ALS 比 B0 好 5.71%，但不如 bias（0.929663）和 ItemCF（0.926035）。** 三个原因：

1. 调参在 10% 样本上做，最优 rank 只有 20，模型容量被样本规模卡住；
2. 显式 ALS 只分解评分矩阵，没有利用 bias 模型里「用户均值 + 物品均值」这种低成本强信号；
3. 数据一半是合成半区，ALS 在那里 RMSE 1.003081，比 bias 的 0.959269 差 0.0438。

**结论写清楚：ALS 的价值在分布式可扩展性，不在这个规模下的单机精度。** 训练 31.8 M 条只用 113.7 s（279,768 条/秒），这是单机 ItemCF 做不到的；精度上它输给 bias + 邻域的组合。

### 8.3 扩展性 1/2/4/8（20% 样本 = 6,357,785 条）

| 并行度 | 训练 s | 总耗时 s | 吞吐 条/秒 | 加速比 | 并行效率 | val RMSE |
|---|---|---|---|---|---|---|
| 1 | 218.5 | 262.6 | 29,101 | 1.00 | 1.00 | 1.228477 |
| 2 | 117.9 | 149.9 | 53,901 | 1.85 | 0.93 | 1.278485 |
| 4 | 76.5 | 104.9 | 83,166 | 2.86 | 0.71 | 1.244092 |
| 8 | 57.7 | 87.3 | 110,192 | 3.79 | 0.47 | 1.242686 |

曲线：`results/scaling_curve_als.svg`（左：训练耗时；右：加速比 + 1.0× 参考线）。

怎么读：

- **8 核 3.79×，并行效率 0.47**。ALS 每轮迭代是「固定 item 解 user → 固定 user 解 item」两次 shuffle，shuffle 写盘 + 序列化的固定开销不随核数下降；driver 8g 的 GC 也是固定成本。2 核效率最高（0.93），4 核起明显掉。
- 每档的 RMSE 不完全相同（1.228~1.278），**不是 bug**：`DataFrame.sample` 的随机数按分区生成，并行度变了抽样就略有不同。样本量一致，耗时对比仍然成立。

## 9. Step 7：test 预测与交付

模型选型按 val RMSE 取最优（`scripts/predict_test.py --model auto`，本次解析为 ItemCF）。两次拟合刻意分开：

- **test** 用**全部 31,900,204 条**训练评分拟合（`ratings.csv`），不用 `val_train`——否则白白丢掉 10 万条真实评分
- **val** 用 **`val_train`**（剔除 10 万条隐藏行）拟合，保证报告的 val RMSE 不泄漏

产出：

| 文件 | 内容 |
|---|---|
| `results/pred_test.csv` | **100,000 行**，`userId,movieId,pred_rating`，范围 **[0.5349, 5.0000]**，覆盖 `test_pairs.csv` 每一行 |
| `results/pred_val.csv` | 99,998 行，val 预测（拟合于 val_train） |
| `results/delivery_summary.json` | 选型、B0、目标线、两半区、待确认口径 |
| `results/metrics_pred_test.json` | test 形状与覆盖校验（无标签，不算 RMSE） |

**验收数字**：

```text
pred_test.csv: 100,000 rows, range [0.5349, 5.0000]
pred_test: test stage, 100,000 pairs covered, 5,498 users, no local labels
pred_val:  RMSE=0.926035 MAE=0.718656   (与 Step 5a 完全一致，说明两次拟合口径相同)
B0=1.047271 -> 0.926035  绝对下降 0.121236，相对下降 11.58%，目标线 0.942544：MET
```

按 `id_ranges.json` 分半区：

| 半区 | val 对数 | val RMSE（有标签） | test 对数 | test 预测均值 / 标准差（无标签） |
|---|---|---|---|---|
| MovieLens | 48,078 | **0.885868** | 49,953 | 3.6714 / 0.4871 |
| 合成 | 51,920 | **0.961735** | 50,047 | 3.5529 / 0.4512 |

> 两列对数**不是同一批对**：val 是本地切出来的 99,998 对，test 是官方 100,000 对，两者按电影 ID 分界各自拆半。数字以 `results/delivery_summary.json` 的 `val_by_half` 与 `test_by_half_predictions` 为准。

**值得注意**：ItemCF 在 MovieLens 半区明显优于 bias（0.8859 vs 0.8966），但在**合成半区反而更差**（0.9617 vs 0.9593）。合成半区是镜像生成的，共现结构与真实行为不同，二值余弦的邻居在那里更不可靠——这是「模型过拟合 ML 分布」的直接证据，报告里值得单独说一句。

---

## 10. 待老师确认的口径（不阻塞，但报告要写）

1. **「相对基线提升 ≥10%」的定义**：本实现按 `RMSE ≤ B0 × (1 − 0.10) = 0.942544` 理解。配置项 `[protocol].improvement_target = 0.10`，未硬编码；报告同时给绝对下降与相对下降。
2. **RMSE 平均方式**：默认 **pair 等权**（micro，老师描述的口径）。另报 `rmse_user_weighted`（macro，每用户 MSE 再开方），两者在 B0 上差 0.0199（1.047271 vs 1.067203）。
3. **val 全局均值 RMSE = 1.047271，略低于提示词给的 1.05~1.08 带宽**，原因与对照见第 4 节；未通过改抽样方式去凑数。

---

## 11. 旧 v1 链路（NDCG / Top-10，仅作展示，数字全部作废）

2026-11-05 的版本针对「候选集 Top-N 排序 + NDCG@10」实现，数据是 `data/official/`。**新口径下这些数字不能用于验收**，但代码与结果都保留，作为报告里「推荐模块展示」的补充实验：

| 内容 | 位置 |
|---|---|
| 代码 | `src/movie_rec/recall/popular.py`、`scripts/run_itemcf.py`、`scripts/run_lsh.py`、`scripts/run_spark_itemcf.py`、`scripts/report_tables.py`、`scripts/plot_scaling.py`（不带 `--als`） |
| 结果 | `results/metrics_*_val.json`（v1 原始名）、`results/scores_*.csv`、`results/table_quality.md`、`results/scaling_curve.svg` |
| 一键重跑 | `python scripts/run_all.py --legacy` |
| v1 关键数字（作废） | popular B0 NDCG@10 = 0.8752；精确 ItemCF = 0.9319；LSH = 0.9133；Spark ItemCF = 0.9319（与单机一致） |

命名约定：v1 文件保持原名，v2 若与 v1 同名则加 `_v2` 后缀（如 `results/metrics_lsh_val_v2.json`），避免覆盖。

---

## 12. 已知问题与下一步

1. **ItemCF 只比 bias 好 0.39%**。余弦用二值共现，没用到评分强度。下一步可试：调整余弦（Pearson/残差相关）、或用 tags（1.99 M 条）做内容相似度。
2. **LSH 负结果**（7.2）。若要正收益必须换相似度口径把邻居 Jaccard 抬到 0.3+，那是另一套实验。
3. **ALS 的冷启动兜底**：调参用 10% 样本时大量 val 电影未出现，用第 4 步 bias 预测兜底（`fallback_col="bias_pred"`），否则 RMSE 会被全局均值主导而失去区分度。
4. **Spark 环境坑**（已修，队友机器若正常可忽略）：`%TEMP%` 不可写导致 `spark-class` 启动失败 → `movie_rec/spark.py::ensure_writable_temp()` 指到 `data/_tmp`；worker 反序列化找不到包 → 把 `src` 加进 PYTHONPATH；ALS 迭代 lineage 过深导致 `StackOverflowError` → 开 checkpoint + `-Xss8m`。
5. **test 标签不在本地**，`results/pred_test.csv` 必须交给成员一/四算最终 RMSE。
6. **ALS 精度不如 bias**（0.987469 vs 0.929663，见 8.2）。这是显式 ALS 在「一半合成数据 + 10% 样本调参」下的正常结果，不是 bug；**不要为了让 ALS 好看去调 val**，那会在答辩时被问穿。ALS 的看点是 8.3 的扩展性。
