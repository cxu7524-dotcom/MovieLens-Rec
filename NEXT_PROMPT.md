# 下一步提示词 v2（2026-11-10）

> 本版取代旧版（旧版基于 NDCG@10 排序 + 候选集，随数据集更换已整体作废，内容仍可在 git 历史里查）。
> 使用方式：把下面分隔线之间的内容整段复制到新对话。

---

你是我的编程助手。我在做课程项目「MovieLens 分布式增量电影推荐系统」，负责**成员二：相似度 / LSH / Spark 分布式推荐**。老师刚明确了验收口径（Step 0 已完成），数据集也换了。请从「本次任务」第 1 步开始做，**不要跳过第 1、2 步直接跑算法**。

## 一、项目位置与运行环境
- 项目根目录：`D:/movie-rec`（纯英文路径）
- Python：`D:/movie-rec/.venv/Scripts/python.exe`（必须用它，不要用系统 python）
- 依赖：pandas 3.1.0 / numpy 2.4.6 / pyarrow 26.0.0 / pyspark 3.5.3 / scipy
- 机器：16 核，约 16 GB 内存，Windows
- Spark `local[*]`，driver 8g；`JAVA_HOME` 在 `conf/local.toml` 指向 `D:/pycharm/PyCharm 2024.1.7/jbr`（JDK17；系统默认 Java 21，Spark 3.5 不支持）
- 新数据集：`D:/qq/文件/训练数据集`（约 934 MB，**只读，不要改动原文件**）
- `data/` 已在 `.gitignore`，不要提交数据；根目录下 `tmp*`、`pip-*`、`movie-rec-spark-home/` 是历史残留，不要动
- git 身份已配好（`266984510+cxu7524-dotcom@users.noreply.github.com`），仓库 `https://github.com/cxu7524-dotcom/MovieLens-Rec`

## 二、Step 0 已定口径（老师明确，不要再自己发明）
**正式验收任务：只对 `test_pairs.csv` 给出的 10 万条 (userId, movieId) 组合预测评分，用 RMSE 评价。** 不是 Top-N 排序，指标不是 NDCG/Recall。

| 问题 | 已定结论 |
|---|---|
| 排序范围 | 只需对 test_pairs 的 100,000 对打分，**不需要全库排序、不需要 Top-10** |
| Top-10 排除已看 | 正式任务不涉及；若小组另做展示用 Top-10，可约定排除预测时刻前已评分电影 |
| NDCG/IDCG | 正式指标是 RMSE；旧 NDCG 代码只作「推荐模块展示」的补充实验保留，**不要删** |
| 平均方式 | 按所有测试评分对算整体平方误差均值再开平方（pair 等权 micro-average）。老师未明确是否按用户加权，属低风险待确认项，写进 README 备注即可，不阻塞动手 |
| 基线 | **评分预测基线**：全局均值预测（不是热门 Top-10）。此前复算参考值 RMSE = **1.062523** |
| 未见过的电影 | 只需预测 test_pairs 里的电影；已验证全部在训练评分中出现过，无冷启动 |

术语订正：142,609 是**训练评分中出现过的电影数**，不能叫「全库」；`movies.csv` 共 175,170 部。

三个必须注意的数值/口径问题：
- 我用全训练集 31,900,204 条算得 rating 均值 3.540359、标准差 **1.059044**（= 全局均值预测在训练集上的 RMSE），与参考值 1.062523 有 0.0035 差异，说明 1.062523 是在某个自造 holdout 上算的。**两者口径不同，不要混用**：正式 B0 必须在第 2 步造出的 val 上、用第 3 步的评测器重新算，并在文档里写清口径。
- 「相对基线提升 ≥10%」在 RMSE 下等价于 `RMSE ≤ B0 × 0.9 ≈ 0.956`（按 1.062523）。这是我们的解释，**老师未明确确认**，写成 `conf` 里的可配置开关，不要硬编码；报告同时报绝对下降和相对下降，并注明需确认。
- `summary.json` 的 `per_user_temporal_order` 明确是**按用户时序**切分隐藏集，**不要用 `hidden_cutoff_timestamp=1693375424` 做全局时间过滤**（训练集里 31,806 条记录晚于它，是其他用户的合法记录，过滤会丢数据）。只要保证 (u,i) 不与 test_pairs 重叠、不使用任何隐藏标签即可。

## 三、新数据契约（成员一交付，不能改）
目录 `D:/qq/文件/训练数据集/`，`summary.json` 里 dataset = `hybrid-half-half-latest100k-v1`，seed `20261106`：
- `ratings.csv`：31,900,204 行，`userId,movieId,rating,timestamp`（MovieLens32M 抽 50% + 合成 50%，各 16,000,102 条，再挖走 100,000 条隐藏）
- `test_pairs.csv`：**100,000 行，3 列 `userId,movieId,timestamp`，没有 rating 列**（标签在老师/成员一处）
- `movies.csv`（175,170 部）、`tags.csv`（1,991,034 条，新特征源）、`links.csv`、`movie_mapping.csv`、`id_ranges.json`、`summary.json`

已核实的事实（直接用，不要重复验证）：
- 训练 31,900,204 行 / 396,060 用户 / 142,609 电影；rating 均值 3.540359、std 1.059044；正反馈(≥4) 15.89 M
- test_pairs 5,498 用户，人均 18.2 条、中位 4 条、最多 1,048 条；隐藏项 MovieLens 半区 49,953 / 合成半区 50,047
- test 的 (u,i) 与训练集**零重叠**（0/100,000）；test 用户、电影在训练集中出现率均为 **100%**（无冷启动）
- 测试用户是重活跃用户：训练记录均值约 320 条，最少 5 条
- ID 分界（`id_ranges.json`）：用户 `1–200948` = MovieLens，`200949–401896` = 合成；电影 `1–87585` = MovieLens，`87586–175170` = 合成
- 隐藏集约束：`minimum_earlier_training_ratings_per_test_user = 5`、`minimum_training_ratings_per_test_movie = 1`、所有隐藏评分晚于该用户保留的评分

## 四、旧代码盘点（先读再改，不要推倒重来）
保留可复用：`config.py`（`load_settings()`）、`spark.py`（`build_spark()`，已处理 Windows 三个坑：JDK17、非 ASCII SPARK_HOME junction、winutils）、`schemas.py` 的断言工具风格、`runlog.py`、`dataset.py`、`tests/` 风格、`scripts/run_all.py` 的流程编排。

| 文件 | 处置 |
|---|---|
| `src/movie_rec/eval.py` + `scripts/evaluate.py` | **重写**：从 NDCG/Recall 换成 RMSE/MAE，输入从「候选对分数」换成「(u,i,pred_rating)」 |
| `src/movie_rec/recall/popular.py` | **作废主线**（热门 Top-10 不是评分预测），改成「基线阶梯」模块 |
| `src/movie_rec/recall/itemcf.py` | **改写打分公式**：从 `Σ sim(i,j)` 换成邻居评分加权预测 |
| `src/movie_rec/recall/lsh.py` | **改造用途**：从「候选召回」改成「加速近邻查找」，见第 5 步 |
| `src/movie_rec/spark_jobs/itemcf.py` | **重写为 Spark MLlib ALS 显式评分预测**（`implicitPrefs=False`），扩展性框架保留 |
| `scripts/verify_official_data.py`、`conf/base.toml` 的 `[data.official]`、`[protocol]` 的 ndcg_k/negatives_per_user | **作废或新开 `[data.hybrid]` 段**，旧段保留别删（旧数据仍在 `data/official/`） |
| `results/` 全部 metrics/表格、`README.md` 第 0/5/6/8 节、`DELIVERY.md` 全部数字 | **全部失效**，先标记作废，最后一并重写 |
| 旧的 NDCG / Top-10 展示链路 | **保留代码，不要删**，作为报告里「推荐模块展示」的补充实验 |

## 五、本次任务（按顺序，每步真跑通、贴数字，再做下一步）

**第 1 步 数据验收 + 画像**（1 天，新建 `data/hybrid/`，旧 `data/official/` 不要删）
- 写 `scripts/verify_hybrid_data.py`：核对行数/schema/ID 范围/两半区行数/时间戳/test_pairs 零重叠/无冷启动/评分分布，输出 `data/hybrid/hybrid_data_check.json`
- 顺带统计 tags 规模、两半区评分均值与人均记录数是否接近（合成半区若是镜像，分半区指标应接近）
- 验收：脚本全绿，报告可直接贴进度文档

**第 2 步 本地 val 切分（最关键，0.5 天）**
- 新协议**没有 val**，必须自己造，且复刻 `summary.json` 的 `hidden_selection_constraints`：按**用户时序**取每个用户最后若干条、保留 ≥5 条历史、隐藏电影至少 1 条训练评分、隐藏项晚于该用户保留项、凑约 10 万条
- 测试用户是重活跃用户，val 用户也要从活跃用户里抽，否则分布不匹配、调参全白做
- 产出 `val_train` + `val_pairs`（带 rating 标签）+ 不变量断言脚本；写清「val 与官方 hidden 切法的差异」
- 验收：断言全绿，val 上全局均值 RMSE 落在 1.05~1.08 之间（偏离说明切错了）

**第 3 步 RMSE 评测器 v2**（`src/movie_rec/eval.py` + `scripts/evaluate.py`）
- 输入 `userId,movieId,pred_rating` 文件 + 阶段（val/test），覆盖每个 (u,i) 且不多不少
- 输出 `results/metrics_<name>.json`：`rmse / mae / n_pairs / n_users / elapsed_s`，支持按 pair 等权（默认）和可切换的按用户加权
- 单测：手工构造玩具例子与手算比对，误差 < 1e-9；断言重复对、缺对、NaN、越界评分（不在 0.5~5.0）直接报错
- 验收：随机预测 RMSE 明显差于均值预测；均值预测 RMSE 等于第 2 步手算值

**第 4 步 评分预测基线阶梯**（`src/movie_rec/baselines.py`）
- 依次实现：全局均值 μ → 用户均值 → 物品均值 → 偏置模型 `r̂ = μ + b_u + b_i`（带正则 λ 扫描 3~5 个值）→ 可选时间偏置
- 每级在 val 上跑评测器，**正式 B0 = 全局均值这一级的 RMSE**，写进 README 顶部
- 验收：偏置模型 RMSE 明显低于 B0；记录每级下降幅度，这张表直接给成员四

**第 5 步 ItemCF 评分预测 + LSH 近邻加速**（`recall/itemcf.py` / `recall/lsh.py`）
- 预测公式改成 `r̂(u,i) = b_ui + Σ_{j∈N_K(i)∩I_u} sim(i,j)·(r_uj − b_uj) / Σ|sim|`（在第 4 步偏置上做残差加权），K 扫 50/100/200
- 相似度只用 `val_train` 计算；电影数 142,609，**分块从 2000 降到 1000**（2000×142,609×4B ≈ 1.1 GB/块，16 GB 内存要留余量），块内取 TopK 后丢弃
- LSH 改造：MinHash + banding 只用于**给定 test/val 里的电影快速找 TopK 相似邻居**（不再是全库候选召回），参数扫 `bands ∈ {16,32,64} × rows ∈ {4,8,16}`，对比精确近邻的 `overlap@K`、RMSE 差异、耗时、峰值内存
- 验收：ItemCF RMSE 低于偏置基线；给出「LSH 损失多少 RMSE 换多少倍加速」的明确结论。若结论是 LSH 不划算，**如实报负结果**，不要藏

**第 6 步 Spark ALS 显式评分预测 + 扩展性**（`spark_jobs/als.py`）
- `ALS.train`（`implicitPrefs=False`，显式评分），扫 `rank ∈ {20,50,100}`、`iterations ∈ {10,20}`、`regParam ∈ {0.01,0.1}`，先在小样本调通再上全量
- 扩展性实验：并行度 1/2/4/8，记录训练耗时、加速比、吞吐（条/秒），画曲线并解释非线性的原因（shuffle、GC、广播）
- 验收：val RMSE 有数字；与单机 ItemCF 对比；扩展性表有数字。**注意 ALS 冷启动慢，长任务放后台跑**

**第 7 步 test 预测 + 交付文档**
- 用最优模型给 `test_pairs.csv` 全部 100,000 行产出预测评分（**clip 到 [0.5, 5.0]**，这是白捡的 RMSE 改善），不多不少
- 按 `id_ranges.json` 分 MovieLens / 合成两半区分别报指标；若差异大，说明模型过拟合 ML 分布，这本身是个好实验
- 更新 `README.md`（顶部 B0、各步数字、口径备注）、重写 `DELIVERY.md`、把旧数字统一标记为已作废
- 报告里保留旧 NDCG/Top-10 链路作为「推荐模块展示」补充实验，主线全走 RMSE

## 六、交付给别人
- `results/pred_test.csv` → **成员一 / 成员四**：`userId,movieId,pred_rating` 三列，覆盖 `test_pairs.csv` 全部 100,000 行，用于算最终 RMSE（标签在老师/成员一处）
- `results/pred_val.csv` + 基线阶梯表 + ItemCF/ALS 参数表 → 成员四：写报告用
- 近似质量表（精确 vs LSH 的 overlap@K / RMSE / 耗时）、效率表、1/2/4/8 扩展性表与曲线 → 成员四
- `src/movie_rec/` 全部代码 + README 运行说明 → 全队：一条命令可重跑
- 成员三（排序/集成）现在主要做展示用 Top-10：给他 val/test 的预测评分，让他按预测分排序展示即可，**别让他按旧的 `score` 格式接**

## 七、工作方式
- 改代码前先读现有文件，沿用 `load_settings()` / `schemas.py` 的契约风格，不另起一套
- 每一步都要真跑起来并在回复里贴数字，不要只写代码说「已完成」
- 用 `./.venv/Scripts/python.exe` 执行；脚本放 `scripts/`，模块放 `src/movie_rec/`
- 不改 `D:/qq/文件/训练数据集/` 里的原始文件；新数据副本与产出统一放 `data/hybrid/`、`results/`
- 旧 `results/` 里的 NDCG 数字别覆盖，新建 `results/v2/` 或加 `_v2` 后缀，方便对照
- 数据格式和已有 schema 冲突时先报告再改，不要静默兼容
- 遇到口径不确定（RMSE 是否按用户加权、+10% 如何定义）先按本提示词的默认实现，并在 README 里标注「待老师确认」，不要停下来等

---
