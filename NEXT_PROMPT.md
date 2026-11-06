# 新对话开场提示词（成员二：相似度与 Spark 推荐）

> 复制下面整段到一个新对话即可继续工作。
> 项目已从中文路径整体移到 `D:\movie-rec`，虚拟环境、配置、数据均已验证可用。

---

你是我的编程助手。我在做一个课程项目（MovieLens 分布式增量电影推荐系统），我负责「成员二：相似度与 Spark 推荐」。仓库已搭好基础设施，官方数据已到位并验收通过。请从「本次任务」第 1 步开始做。

## 一、项目位置与运行环境

- 项目根目录：`D:/movie-rec`（纯英文路径，已从原中文目录整体迁移）
- Python 解释器：`D:/movie-rec/.venv/Scripts/python.exe`（**必须用它，不要用系统 python**）
- 依赖：pandas 3.1.0 / numpy 2.4.6 / pyarrow 26.0.0 / pyspark 3.5.3
- 机器：16 核，约 16GB 内存，Windows
- Spark 用 `local[*]`，driver 8g；`JAVA_HOME` 在 `conf/local.toml` 指向 `D:/pycharm/PyCharm 2024.1.7/jbr`（JDK17；系统默认是 Java 21，Spark 3.5 不支持）
- `data/` 已在 .gitignore 里，不要提交数据
- 项目根目录下有几个 `tmp*` 目录，是历史运行残留，不要动

## 二、已有的代码（先读再改）

```
README.md                 # 进度、验收命令、本机环境说明
conf/base.toml            # 共享配置；[data] 是开发样本，[data.official] 是官方数据
conf/local.toml           # 本机覆盖（JDK 路径，git 忽略）
src/movie_rec/config.py   # load_settings() -> Settings(data, official, protocol, spark)
src/movie_rec/schemas.py  # Parquet/Spark schema + 断言工具（含官方数据契约）
src/movie_rec/spark.py    # build_spark()，已处理 Windows 三个坑（JDK、非ASCII SPARK_HOME、winutils）
scripts/check_env.py            # 工具链与配置检查
scripts/make_dev_fixture.py     # 用 ml-latest-small 造开发样本（产物在 data/dev/）
scripts/validate_fixture.py     # 开发样本不变量校验
scripts/smoke_test.py           # 环境冒烟（Spark 冷启动慢，建议后台跑）
scripts/verify_official_data.py # 官方数据验收，已全绿
data/official/                  # 官方数据（已解压，1.3GB）
data/official/official_data_check.json  # 验收报告
```

`src/movie_rec/` 目前只有上面三个模块，**没有任何召回算法**。

## 三、官方数据契约（成员一交付，不能改）

路径都在 `conf/base.toml` 的 `[data.official]`，CSV 格式：

| 文件 | 列 | 说明 |
|---|---|---|
| `processed/train.csv` | userId, movieId, rating, timestamp | 训练用；每位用户自己的记录都已截断在其推荐时刻之前 |
| `processed/validation.csv` | userId, movieId, rating, timestamp | 验证标签，rating≥4 为正反馈 |
| `raw/ml-32m/movies.csv` | movieId, title, genres | 类别特征 |
| `candidates/validation_candidates.csv` | userId, movieId | 验证阶段候选，**无标签列** |
| `candidates/test_candidates.csv` | userId, movieId | 测试阶段候选，**标签在成员一手里** |
| `candidates/validation_cutoffs.csv` | userId, cutoff_timestamp | 推荐时刻 |
| `candidates/test_cutoffs.csv` | userId, cutoff_timestamp | 推荐时刻 |

**已验证的事实（不要重复验证，直接用）**：

- train 25,520,897 行 / 200,948 用户 / 71,288 电影 / 正反馈 12,905,652
- validation 3,188,339 行 / 正反馈 1,472,848
- val 候选 20,281,548 行 / 188,087 用户，每人 101–652 个（中位 104）
- test 候选 20,660,431 行 / 191,007 用户，每人 101–653 个（中位 105）
- 每人候选数 = 该阶段正样本数 + 100 个负样本
- 所有用户的训练记录都早于自己的 cutoff（0 例越界），**自家历史不需要再过滤**
- 验证阶段 100% 的正样本都在验证候选集内
- 两阶段共 4094 万条 (userId, movieId) 待打分

**必须遵守的规则**：
1. 用同一套划分和候选集，不重新抽样；随机种子 20260907。
2. 不得把 `validation.csv` 里该用户的评分当作已知历史（它是要预测的）。
3. test 阶段没有标签，只能产出分数交给成员三。
4. 记录数据版本、参数、种子、环境。

## 四、本次任务（按顺序做，每步跑通再做下一步）

**第 1 步：本地评测器**（`src/movie_rec/eval.py` + `scripts/evaluate.py`）
- 输入：`userId,movieId,score` 的分数文件 + 阶段（val/test）
- val 阶段：把候选集与 `validation.csv`（rating≥4）左连接得到标签，算 NDCG@10、Recall@10、覆盖率、单用户平均延迟
- 输出 `results/metrics_<name>.json`，字段：`ndcg@10 / recall@10 / coverage / users / elapsed_s`
- 写单测：手工构造 3 个用户的玩具例子，与函数输出对比，误差 < 1e-9
- 验收：跑一个随机分数的假预测不报错，且 NDCG 数值明显低于 0.1

**第 2 步：热门基线**（`src/movie_rec/recall/popular.py`）
- 全局热门 Top10（按 train 正反馈次数）+ 时间衰减热门（`score = Σ exp(-λ·(t_cutoff - t_i))`，λ 试 3 个值）
- 给 val 候选集每一对打分，产出 `results/scores_popular_val.csv`
- 跑第 1 步的评测器，得到基线数字 **B0**
- 验收：B0 写进 `README.md` 顶部，作为全队及格线

**第 3 步：精确 item-item 相似度**（`src/movie_rec/recall/itemcf.py`）
- 用 train 正反馈建 user×item 稀疏矩阵（scipy.sparse）
- item-item 余弦相似度：**必须按行分块**（每块 2000 部电影），块内取 TopK（K=50/100/200）后丢弃块外结果
- 用户打分：`score(u,i) = Σ_{j∈I_u} sim(i,j)`，只对候选集里的 (u,i) 取值
- **禁止**一次性算 71288×71288 稠密矩阵（约 20.3GB）
- 产出 `results/scores_itemcf_val.csv`，跑评测，与 B0 对比
- 验收：NDCG@10 明显高于 B0；内存峰值 < 8GB

**第 4 步：LSH 近似召回**（`src/movie_rec/recall/lsh.py`）
- MinHash + banding（近似 Jaccard），只吃二值化矩阵（**不能用于评分加权**）
- 参数扫至少 6 组：`bands ∈ {16,32,64} × rows ∈ {4,8,16}`
- 记录：与精确 TopK 的 overlap@100、候选生成耗时、峰值内存
- 产出参数表 + `results/scores_lsh_val.csv`
- 验收：给出「LSH 损失 X% 召回率换来 Y 倍加速」的结论

**第 5 步：Spark ItemCF**（`src/movie_rec/spark_jobs/itemcf.py`）
- 分布式共现矩阵 → 相似度 → topK 截断 → 广播打分
- 先用 `local[*]` 在 val 候选上跑通，再上 test 候选
- 扩展性实验：`spark.master` 并行度 1/2/4/8，记录总耗时、加速比、吞吐
- 验收：与单机精确版在同一候选集上的 NDCG 差异 < 0.001；扩展性表有数字

## 五、交付给其他人

| 交付物 | 给谁 | 格式 |
|---|---|---|
| `results/scores_itemcf_val.csv` `scores_lsh_val.csv` | 成员三 | `userId,movieId,score` 三列，**覆盖候选集每一对**，没打分的填 0 不要丢 |
| `results/scores_itemcf_test.csv` `scores_lsh_test.csv` | 成员三 | 同上 |
| 近似质量表（精确 vs LSH 各参数） | 成员四 | NDCG@10 / Recall@10 / overlap@100 |
| 效率对比表 | 成员四 | 耗时、内存、单用户延迟 |
| 扩展性表 + 曲线 | 成员四 | 1/2/4/8 并行度的耗时与加速比 |
| `src/movie_rec/recall/` 全部代码 + README 运行说明 | 全队 | 一条命令可重跑 |

## 六、工作方式要求

- 改代码前先读现有文件，沿用已有的 `load_settings()` / `schemas.py` 契约，不要另起一套。
- 每一步都要真的跑起来并把数字贴给我，不要只写代码说「已完成」。
- 用 `./.venv/Scripts/python.exe` 执行，脚本写在 `scripts/`，模块写在 `src/movie_rec/`。
- Spark 冷启动慢，长任务放后台跑。
- 不要动 `data/official/` 里的原始文件，产出统一放 `results/`。
- 遇到数据格式和已有 schema 冲突时，先报告再改，不要静默兼容。
