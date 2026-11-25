# 100分钟预算的真实静态 ADMTune 实验

日期：2026-11-25。运行目录 `timed-static-100min-002`；独立容器 `autodl-container-aa6147b8a1-c2c6efb0`；源码 `26687eb4259e56c439edddc7e568cddd21b636d0`。

## 1. 协议与完成情况

- MySQL 8.0.36，独立3308实例，四张 Sysbench 表、每表200万行；运行前从同一不可变冷快照恢复。
- 原生 `oltp_read_only.lua`，uniform 随机访问，32线程，种子42；每个配置15秒预热、60秒正式测量。不是动态负载，也不是 TPC-H。
- 调优墙钟预算100分钟，包括真实模型调用、GP拟合、执行、周期复测和最终配置恢复；冷恢复在计时前完成。实际活动耗时 **97.81分钟**，最后观测完成于 **97.80分钟**。不把剩余预留时间伪造为试验点。
- **71次真实试验、71次有效观测、26种配置**。100分钟不等于100次；本序列未拼接同容器旧22次记录或另一容器最后核实的14次记录。
- 停止原因 `time_or_api_budget`；真实末端复测完成、初始配置恢复及数据库四表行数、逻辑样本哈希复核通过。
- 真实模型 `qwen3.7-plus`，**13次API、23187 tokens**，没有启发式降级。全局40次调用/160000 tokens为上限，不是实际支出。

| 观测类型 | 次数 |
|---|---:|
| `baseline` | 1 |
| `startup` | 4 |
| `audit` | 7 |
| `incumbent_repeat` | 37 |
| `candidate` | 14 |
| `reference` | 6 |
| `verification` | 2 |

`incumbent_repeat` 是已有配置噪声重复，不是新增候选或搜索收益；`reference` 是初始配置穿插复测；`verification` 是独立检查点/末端复测。

## 2. 核心结果

| 指标 | 初始配置 | 末端独立复测 |
|---|---:|---:|
| Throughput (TPS) | 2449.19 | 3807.49 |
| Latency (ms，事务P95) | 18.28 | 68.05 |

- 相对初始TPS增益 **55.46%**；相对穿插默认配置中位数增益 **56.24%**（默认复测中位数 2437.00 TPS）。
- 搜索期间最高新配置观测 **3931.63 TPS**，第 46 次，时间 63.36 分钟。该点与末端独立复测分开，不做“取最大值复测”。
- 优化目标固定为TPS最大化，不是TPS/P95；P95由18.28 ms升至68.05 ms，明确变差，不能写成吞吐与尾延迟同时改善。若论文目标包含低尾延迟，应先冻结延迟约束或双指标目标，再另起独立运行，不能事后改本轮优化目标。
- 最终接受配置：`{"innodb_buffer_pool_size": 4294967296, "innodb_io_capacity": 1200, "innodb_io_capacity_max": 4500, "sort_buffer_size": 262144}`。

## 3. GP、增量反思及预测检验

最终GP使用 **71条**本运行有效观测；输入为四参数完整编码加常量静态上下文，目标为原始 `log(TPS)`。静态运行不能证明动态上下文建模或跨阶段迁移收益。当前主核为联合 Matérn-5/2；不能写成已经运行“配置Matérn核+上下文线性核”的主模型。

最终超参数 `{"amplitude": 1.394314430351689, "length_scale": 0.4820525324708844, "noise_std": 0.13423705645741035}`；优化状态及维度见 `final_gp.json`。每次前瞻预测均从试验前快照重建核验，训练ID严格早于当前观测，没有把当前结果回填为预测。

| 被选中观测类型 | 前瞻预测数 | MAPE | ±2观测标准差覆盖率 |
|---|---:|---:|---:|
| `audit` | 6 | 5.36% | 83.3% |
| `incumbent_repeat` | 37 | 1.63% | 78.4% |
| `candidate` | 14 | 3.46% | 100.0% |
| `reference` | 6 | 1.37% | 83.3% |
| `verification` | 2 | 1.21% | 100.0% |

点预测采用 `exp(mean_log)`；超参数每10个模型版本优化一次，其他拟合复用最近超参数。覆盖率与误差只描述在线被选择的数据；重复配置、稀疏抽样审计与真正新候选分别列出，不能把重复点的高覆盖率当作全空间校准证明。

共 **7次增量反思**，关联70条记录；证据编号及事实与原始观测一致。单次静态运行不用于证明多时间片检测、历史跨负载复用或反思消融收益。

## 4. 图与数据来源

- `docs/figures/admtune_100min_measured.png/.pdf`：同协议的 ADMTune 与 Default，吞吐/延迟两面板；末端星号为独立复测，灰叉为初始配置穿插复测。
- `docs/figures/admtune_100min_comparison.png/.pdf`：六方法吞吐图，采用 AgentTune 风格，横轴 Tuning Time (min)。ADMTune为本次真实观测的best-found曲线；Default为本次初始配置实测水平参考线，不是补造观测。
- AgenticDB、AgentTune、GPTuner、CDBTune来自用户提供 AgenticDB 原文第9页图5 `MS Sysbench Read` TPS子图，直接抽取PDF矢量折线及刻度坐标，不从表5的TPS/P95目标值冒充TPS，不推造原文试验次数。
- 原文 AgenticDB 曲线保留实际终止时间；不延长为未提供的100分钟观测。表中数值、单位、时间坐标与来源均存入 `literature_curves.csv` 和 `literature_provenance.json`。
- 这些原文曲线不是本服务器重跑结果：原文含OS参数、目标为TPS/P95，第5.2节原文环境为MySQL 8.0.45/64 GB/GPT-5.5、三次重复取中位运行；本次为MySQL 8.0.36/90 GiB容器/qwen3.7-plus、单次运行，调参空间也不同。六线图是**跨实验参照**，不能据此宣称同条件排名。不能仅因CPU核数相同就认为整个协议相同；正式主实验需要同协议重跑基线。

## 5. 留档、续接和复现

- 原始日志、SQLite账本、模型快照、模型请求/用量及恢复审计完整存入本地忽略目录 `runs/collected-timed-static-100min-002/`。
- 私有归档 `runs/archives/timed-static-100min-002.tar.gz`，SHA256 `1fdbb5619a3529ee4a1dc16839aa6e801cf1fe5cbf64b343ce0eab94ae4c1c86`；归档含配置凭证，不提交远程。
- 可公开的数据在 `docs/experiment_data/timed_static_100min_002/`：`admtune_trials.csv`、`predictions.csv`、`verification.json`、`final_gp.json`、`literature_curves.csv`、`literature_provenance.json`、`comparison_curve.csv`。图线保留真实耗时、独立终点与数据来源。
- 当前累计编号71，下一编号72；可按冻结清单继续编号，但后续100分钟会话是额外预算，不能并入首个100分钟图。
- 下一步是同协议基线、独立种子重复与动态/消融，而不是把一条100分钟轨迹包装成完整论文矩阵。

旧22条及未完成第23条未改写，已额外备份为本地忽略目录中的 `runs/archives/pre-time100-002-old23.tar.gz`；本轮数据库恢复前后的审计也已留档。

绘图：在安装 matplotlib 的环境运行 `python docs/analysis/plot_timed_static.py`。原文抽取脚本 `docs/analysis/extract_agenticdb_fig5.py --pdf <用户提供的AgenticDB.pdf> --output docs/experiment_data/timed_static_100min_002` 需要 pdfplumber；PDF固定SHA校验，不从网上猜测原文坐标。

独立复核：`python docs/analysis/verify_timed_static.py --output runs/reverified-timed-static-100min-002`。仅使用已有私有归档，不连接数据库或LLM；输出目录必须是新的，避免覆盖证据。
