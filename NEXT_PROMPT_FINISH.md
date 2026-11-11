# 收尾执行提示词（2026-11-11）

> 用途：成员二 v2 主体已完成并通过复核，只剩「推送 → 补 ALS 扩展性 → 修三处文档」。
> 把下面分隔线之间的内容整段复制到新对话执行。

---

你是我的编程助手。项目「MovieLens 分布式增量电影推荐系统」成员二的 v2 工作（RMSE 评分预测）主体已完成，数字已独立复核无误。现在只做三件收尾的事，**按 1 → 2 → 3 顺序执行，不要重跑已完成的模型**。

## 一、环境与当前状态（已验证，不要重复盘）

- 项目根：`D:/movie-rec`；解释器：`D:/movie-rec/.venv/Scripts/python.exe`（不要用系统 python）
- 依赖：pandas 3.1.0 / numpy 2.4.6 / pyarrow 26.0.0 / pyspark 3.5.3 / scipy 1.17.1
- 机器 16 核 / 17 GB，Windows；Spark `local[*]`，driver 8g；`JAVA_HOME` 在 `conf/local.toml` 指向 `D:/pycharm/PyCharm 2024.1.7/jbr`（JDK17）
- 数据：`data/hybrid/`（31,900,204 条训练评分、99,998 对 val、100,000 对 test）；`data/official/` 是 v1 旧数据，**不要动**
- 仓库：`https://github.com/cxu7524-dotcom/MovieLens-Rec`，当前 HEAD `dfa2062`（v1 版本）
- git 身份已配：`266984510+cxu7524-dotcom@users.noreply.github.com`；gh CLI 在 `/c/Program Files/GitHub CLI/gh.exe`，已登录
- **66 个文件、约 13.6 MB 未提交**，全部是 v2 成果

已复核的基准数字（**任何改动都不得让这些数变化**）：
- B0 全局均值 val RMSE **1.047271**；主交付 ItemCF(K=200,β=20) val RMSE **0.926035**（相对 +11.58%）
- `results/pred_test.csv` 100,000 行、零缺失/零多余/零重复、范围 [0.5349, 5.0]
- val 切分无泄漏：`val_train` 与 `val_pairs` 零重叠，donor 与 5,498 个 test 用户零重叠
- 单测 **36 passed**

## 二、第 1 步：提交并推送（先做，队友现在拉不到任何 v2 内容）

1. 先确认没有大文件混进来：`git add -A` 后跑
   `git diff --cached --name-only | xargs -r du -k | sort -rn | head -10`
   单个文件必须 < 100 MB。预期最大是 `results/pred_test.csv`（约 2.2 MB）。`.gitignore` 已放行 `results/pred_*.csv` 和 `results/v2_table_*.csv`，忽略 `results/*.csv` 的 3 GB v1 分数文件，**不要改这条规则**。
2. 提交信息（中文或英文均可，说清是 v2）：
   `feat(v2): RMSE rating prediction - baselines, ItemCF, LSH negative result, pred_test`
   正文列出：B0 1.047271 / ItemCF 0.926035 / +11.58% / 100,000 行 pred_test / 36 tests。
3. `git push origin main`（凭据管理器已存，不会问密码）。
4. 验收：`git status -sb` 干净；用 gh 核对远端有 `results/pred_test.csv` 和更新后的 `README.md`、`DELIVERY.md`。
5. 失败预案：若被拒先 `git pull --rebase` 再推；若报大文件超限，回头检查第 1 条。

## 三、第 2 步：补 ALS 全量 + 扩展性（任务三的分数缺口，必须补）

**背景**：ALS 目前只跑了 8/12 组 10% 样本调参（`results/als_val_partial.json`），全量 val RMSE 没有，1/2/4/8 扩展性表和曲线完全没有。任务书任务三要求「至少一种推荐算法的分布式实现」+「数据规模扩展性实验」，这是缺的那块。

1. 先看参数：`./.venv/Scripts/python.exe scripts/run_als.py --help`，**按实际 CLI 传参，不要猜**。
2. 跑全量：`./.venv/Scripts/python.exe scripts/run_als.py`
   - 这会补完 12 组调参、用最优配置在全量 31,800,204 条上拟合、产出 `results/pred_val_als.csv` 和 val 指标
   - **预计 30~50 分钟，必须后台跑**，不要前台等到超时
3. 跑扩展性：`./.venv/Scripts/python.exe scripts/run_als.py --scaling`（1/2/4/8 并行度，记录耗时/加速比/吞吐）
4. 生成表与图：`./.venv/Scripts/python.exe scripts/report_v2.py` 然后 `./.venv/Scripts/python.exe scripts/plot_scaling.py --als`
5. 验收（四个都要有）：
   - 全量 ALS 的 val RMSE 是一个具体数字，写进 README 第 8 节和 `v2_table_models.md`
   - `results/v2_table_scaling.md` / `.csv` 存在，含 1/2/4/8 四行（耗时、加速比、吞吐）
   - `results/scaling_curve_als.svg` 存在
   - `v2_table_params.md` 里 ALS 表格不再是空表头
6. 失败预案：
   - Spark 起不来：跑 `scripts/check_env.py`、`scripts/diag_spark.py`
   - `StackOverflowError`：README 第 12 节说已用 checkpoint + `-Xss8m` 修过，确认仍生效
   - `%TEMP%` 不可写：`movie_rec/spark.py::ensure_writable_temp()` 已指到 `data/_tmp`
   - 内存吃紧：把 driver 从 8g 降到 6g，或减小 `--scaling` 的样本
7. **结果如实报**：ALS 打不过 ItemCF（0.926035）是正常结论（显式评分 + 冷启动兜底），不要为了好看去调 val；也不要因此删掉 ALS。报告里写清「ALS 的价值在分布式可扩展性，不在单机精度」。

## 四、第 3 步：修三处文档错误

1. **README 第 9 节分半区对数写错了**：表头是「val RMSE（有标签）」，但对数用了 test 的 49,953 / 50,047。val 实际是 **48,078（ML）/ 51,920（合成）**，以 `results/delivery_summary.json` 的 `val_by_half` 为准。test 的 49,953 / 50,047 是另一列，别混。
2. **`v2_table_params.md` 的 ALS 表**：第 2 步跑完 `report_v2.py` 应自动补；若没补，手工按 `results/als_val_full.json`（或实际产物名）填，不要留空表头。
3. **`DELIVERY.md` 引用的两个文件**：`results/v2_table_scaling.md`、`results/scaling_curve_als.svg` 目前不存在，第 2 步跑完就有了；跑完逐个 `ls` 确认存在，路径名对不上就改 DELIVERY.md 里的引用，不要留死链。

## 五、做完之后再推一次

1. 重跑 `./.venv/Scripts/python.exe -m pytest tests -q`，必须仍是 36 passed（补 ALS 不应改动已有逻辑）。
2. 复核三个数字没变：val ItemCF 0.926035、B0 1.047271、`pred_test.csv` 仍 100,000 行且覆盖零缺失。
3. `git add -A` → 提交（如 `feat(v2): complete ALS full-train + 1/2/4/8 scaling, fix doc numbers`）→ `git push origin main`。
4. 最后回一份清单：ALS 全量 val RMSE、扩展性四行数据、三处文档修了哪几行、远端提交号。

## 六、边界（不要做）

- 不要重跑 ItemCF / LSH / baselines（数字已定，重跑可能因并行或内存差异变动，会污染报告）
- 不要改 `data/hybrid/val_pairs.parquet`、`val_train.parquet`（改了所有数字作废）
- 不要动 `data/official/`、`results/scores_*.csv`（v1 展示用，3 GB，已被 gitignore）
- 不要为了提升数字去反复调 val（过拟合 val 会被答辩问死）
- 不要改 `.gitignore` 里忽略 `results/*.csv` 的规则
- 每步真跑、贴真实输出，不要只改文档说「已完成」

---
