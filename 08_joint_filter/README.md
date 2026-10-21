# Stage6 联合清理

在 Stage5 上复用 LID、heuristics、similarity、QE 和可选 LLM 分数。工作流与
`06_frontier_llm_as_judge` 一致：**先收集统计、生成多组 thresholds，再单独应用选定的一组**。
不调用评分模型、不训练新模型、不提交训练实验。

## 快速运行

从仓库根目录进入本目录。以下提交脚本可从任何目录调用；传入的相对路径按调用目录解析。
Python 默认使用仓库 `.venv/bin/python`（存在时），可通过 `--python /absolute/path/python`
指定计算节点可用的解释器。依赖见 `requirements.txt`。
四个阶段均通过 `submit_*.sh` 提交 SLURM；登录节点只执行提交命令。
四个阶段的默认账户均为 `project_462001249`；使用其他账户时需在各提交命令中显式传入 `--account`。
过滤和汇总提交脚本的默认输出目录为
`/scratch/project_462001509/members/zihao/FineOPUS/FineOPUS-Filtered-Stage6`，
可通过 `--out-dir` 覆盖；存储目录与 SLURM 计算账户独立设置。
实际提交时 stdout 仅输出 job ID，完整 sbatch 命令写到 stderr，可直接捕获 job ID 设置依赖。
提交器向 worker 显式传入 Python 入口文件的绝对路径，避免 Slurm spool 目录改变脚本位置。
请通过 `submit_*.sh` 提交；worker 的两个位置参数依次为 Python 解释器和 Python 入口文件。

```bash
cd /scratch/project_462001427/FineOPUS/08_joint_filter

# 1. 仅提交统计任务：抽样拟合变换，然后全量扫描，保存每语言对分数直方图。
COLLECT_JOB=$(bash thresholds/submit_collect_data_stats.sh \
  --tasks 64 \
  --account project_462001249 \
  --stats-dir "$PWD/thresholds/stats/stage5_v1" \
  --sample-rows 50000 \
  --token-mode estimated)
```

默认输入为 `/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage5`，默认 token 表为
`../tools/token_stats/FineOPUS-Filtered-Stage5_Qwen3_5.csv`。必须与输入每语言对的行数匹配。
首次执行前可在提交命令末尾加 `--dry-run` 查看完整 sbatch 命令。

提交阈值计算任务，通过 `afterok` 自动等待整个统计数组成功。此步骤只读取小型统计文件，不扫描原始语料：

```bash
THRESHOLDS_JOB=$(bash thresholds/submit_compute_thresholds.sh \
  --account project_462001249 \
  --dependency "afterok:${COLLECT_JOB%%;*}" \
  --stats-dir "$PWD/thresholds/stats/stage5_v1" \
  --profiles 'relaxed=1.2e12,balanced=1e12,strict=8e11')
```

得到：

```text
thresholds/stats/stage5_v1/
  run.json
  pairs/<lang_pair>/
    calibration.json
    histogram.npz
    summary.json
  profiles/
    thresholds_relaxed.csv
    thresholds_relaxed.json
    thresholds_balanced.csv
    thresholds_balanced.json
    thresholds_strict.csv
    thresholds_strict.json
```

CSV 和同名 JSON 是一组；JSON 记录规则、统计快照和 CSV 校验信息。也可增加新目标：

```bash
bash thresholds/submit_compute_thresholds.sh \
  --stats-dir "$PWD/thresholds/stats/stage5_v1" \
  --profiles 'medium=1.1e12,aggressive=9e11'
```

选择一组后，单独提交过滤：

```bash
FILTER_JOB=$(bash filter/submit_apply_thresholds.sh \
  --tasks 64 \
  --account project_462001249 \
  --dependency "afterok:${THRESHOLDS_JOB%%;*}" \
  --thresholds-csv "$PWD/thresholds/stats/stage5_v1/profiles/thresholds_balanced.csv" \
  --out-dir /scratch/project_462001509/members/zihao/FineOPUS/FineOPUS-Filtered-Stage6 \
  --count-tokens)
```

`--count-tokens` 在写出时用本地 `../slm_from_scratch/tokenizer/qwen3_5/tokenizer.json`
对保留样本精确计数，不使用 special tokens。省略该参数会输出估计 token 数，速度更快。
不同阈值组必须写入不同输出目录，不会覆盖已经完成的数据。

提交汇总任务，自动等待全部过滤任务成功：

```bash
bash filter/submit_merge_filter_stats.sh \
  --account project_462001249 \
  --dependency "afterok:${FILTER_JOB%%;*}" \
  --out-dir /scratch/project_462001509/members/zihao/FineOPUS/FineOPUS-Filtered-Stage6
```

输出根目录得到 `filter_stats.csv`、`filter_stats.json`、`filter_stats_corpora.csv`、
`filter_stats_languages.csv`。语言表是该语言关联的双语对 token 总量，包含双侧，不能将各语言行求和
作为整个语料的总量。中途查看进度可独立提交汇总任务并加 `--allow-partial`，结果明确列出缺失语言对。
上游任务已完成时可省略 `--dependency`；单独提交时也可使用实际 job ID，例如 `--dependency afterok:123456`。

## Token 预算的含义

默认 `estimated` 模式：**全量精确统计分数桶的行数，但桶内 token 数是估计值**。
分别累计源侧、目标侧字符数，然后用该语言对现有的真实源侧/目标侧总 token 数缩放。
因此输入总量匹配已有 CSV，但删除后 token 数会因 token/字符比变化产生偏差。
`1T` 是预计目标，不能保证第一次落盘就在 ±1% 内；`--count-tokens` 会报告真实结果。

需要更精确的预算，可在统计阶段直接使用：

```bash
bash thresholds/submit_collect_data_stats.sh \
  --tasks 64 \
  --stats-dir "$PWD/thresholds/stats/stage5_exact_v1" \
  --token-mode exact \
  --tokenizer-json "$PWD/../slm_from_scratch/tokenizer/qwen3_5/tokenizer.json"
```

`exact` 模式不需要 token CSV，对**全部输入文本**执行 CPU tokenization，成本明显更高。
过滤时自动沿用同一个 tokenizer，报告精确的保留 token 数。
两种模式都在边界桶使用稳定哈希抽取指定比例，因此是预算的期望值，而非精确到单 token 的背包求解。
小语言对或单行很长时会有离散误差；如需改预算，使用新的 profile 名称生成阈值并写新输出目录。
不会为了达到目标而恢复硬规则删除或缺少全部语义证据的样本。

## 联合分数

配置集中在 `config.json`。更改配置需要新的统计目录；只修改 token 目标不需要重扫。

1. 使用 NFC 和空白规范化判断复制；源/目标为空直接删除。
2. 跨文字系统复制：至少 20 个字母、文本相同、标注文字系统明显不符时删除。
3. 错语种：至少 40 个字母、两套 LID 同时以 ≥0.90 置信度指向同一个其他语言，且标注文字系统
   的字母占比 <20% 时删除。未知文字系统不作为删除证据。
4. similarity 和 QE 都缺失/非有限时排除，单独记录 `missing_semantics`。LLM 存在也不绕过此规则。
5. 在剩余抽样数据上拟合语言对的 similarity 分位数 P、按 QE 模型分组的分位数 Q、LLM 分位数 J。
   抽样是在整个语言对的全局行号上均匀无放回抽样，覆盖所有 shard 和 row group，无前缀偏置。
6. 计算 LID 一致性 L，以及长度失衡、复制、重复、数字、结构五组 heuristics 风险，合成为 H。
7. 按 P/Q/H/L = 0.35/0.35/0.20/0.10 做加权几何平均。MRR <0.8 时 P 权重乘0.5，
   无 benchmark 乘0.75。QE 任意一侧不支持时 Q 权重乘0.5；状态未知乘0.75。
8. 有 LLM 时，其权重为0.20，归一化后的基础分数组合权重为0.80。
   至少两组有效分数 <0.20 时总分乘0.5；复制风险≥0.90且 L<0.40时再乘0.25。

修正后的字符比/词长使用浮点计算，并从目录目标语言识别无空格文字系统；不读取旧的截断比值和
受 `trg_lang` 列名问题影响的词长字段。输入列原样保留；修正特征只在评分过程中使用。
源/目标最大词长沿用已有精确值。缺失的 ancillary heuristics 不填成“坏样本”；常数校准特征禁用。
没有数字时数字风险为0；短于12个字母的实体不施加复制软惩罚。标点风险按文本长度缩放。
`filter_regex_src` 的用户自定义正则含义未知，不作为自动删除或降分依据。
语言别名映射可在配置中修改；汉字的 Hans/Hant/Hani 按同一文字系统比较，但不将粤语合并成普通话。

分位数以最多257个节点表示，重复值取中间秩。分数仅用于排序，不是质量概率；不同语言对不直接按
原始总分竞争。QE 模型在抽样中未出现时回退到语言对 QE 分布；完全无校准值时使用中性分位数0.5。

## 多组阈值如何产生

每语言对将联合分数分为10,000桶，分别统计有/无 LLM 的两个覆盖组。初始预算权重：

| Stage5 单语言对 tokens | 保留比例权重 |
| --- | ---: |
| <0.1B | 90% |
| 0.1B–1B | 80% |
| 1B–10B | 61% |
| ≥10B | 60% |

对所有语言对统一求系数 `c`，分配 `min(可选tokens, c × 原始tokens × 比例权重)`，使总量达到
profile 的目标。若可选量不足，全部保留可选数据，并在 profile JSON 报告 shortfall。
同一语言对有/无 LLM 两组采用相同的 token 保留比例，避免缺失本身形成排名惩罚。

每个 CSV 语言对有两行，`llm_present=0/1`；`T` 是可读的边界分数。
**实际应用同时使用 `boundary_bin` 和 `boundary_fraction`**：
高于边界桶全部保留，低于边界桶删除，桶内用输入文件名、行号和 seed 的稳定哈希抽取。
这允许处理离散 LLM 分数和大量同分样本，且不受 batch size、SLURM 任务数量影响。
因此不能将 CSV 直接传给旧的 `06/.../filter`；必须使用本目录的联合过滤脚本。

## 运行与续跑约束

- 四个提交器支持 `--tasks/--account/--partition/--cpus/--mem/--time/--dependency/--python/--dry-run`。
  其余参数传给 Python worker；提交器同时接受 `--scored-dir` 和 `--scored_dir`。
- 统计收集和过滤默认64个数组任务、每任务4 CPU/32G、2天。任务按语言对确定性分配；一个大语言对由一个任务处理。
  阈值计算为单任务1 CPU/16G/2小时，汇总为单任务1 CPU/8G/2小时。
  阈值计算和汇总不创建数组，拒绝 `--tasks` 大于1，避免多个进程同时写同一结果。
- 统计阶段：读取抽样所触及的 row group，再做一次全量直方图扫描；内存上限受一个 row group、
  sample size 和 batch size 影响。过滤阶段独立流式读取一次原始数据。
- `--pairs eng_Latn-fra_Latn,abk_Cyrl-eng_Latn` 可对指定语言对生成独立 pilot 统计；需要独立统计目录，
  并在计算阈值时指定适合这个子集的 token 目标。正式全量统计不要传 `--pairs`。
- shard 读取失败即失败，不跳过、不标记完成。缺失阈值、统计任务未完成、token CSV 行数不符均报错。
- 统计记录输入文件名、大小、mtime、行数和 schema；过滤验证快照未变。这不是完整输入内容哈希。
- 输出以语言对为单位写入临时目录，校验后原子发布，并写 `_DONE.json`。
  收集与过滤提交脚本默认 `--skip_existing`；只跳过配置与文件检查匹配的完成项，失败语言对从头重做。
  强制杀进程可能留下隐藏临时目录，续跑不将它们视为完成数据。
- 复用其他 profile 的输出目录、嵌套输入/输出目录、复用不同配置的统计目录均拒绝。
- 不自动调整规则、启动训练、重跑其他 Stage，也不删除 Stage5。

## 提交前检查

```bash
bash thresholds/submit_collect_data_stats.sh --dry-run
bash thresholds/submit_compute_thresholds.sh --dry-run
bash filter/submit_apply_thresholds.sh \
  --thresholds-csv "$PWD/thresholds/stats/stage5_v1/profiles/thresholds_balanced.csv" \
  --out-dir /scratch/project_462001509/members/zihao/FineOPUS/FineOPUS-Filtered-Stage6 --dry-run
bash filter/submit_merge_filter_stats.sh \
  --out-dir /scratch/project_462001509/members/zihao/FineOPUS/FineOPUS-Filtered-Stage6 --dry-run
```

`--dry-run` 只打印提交命令，不运行 Python 数据处理脚本，也不向 SLURM 提交作业。
