# 100分钟真实静态实验：最后核实状态

运行日期：2026-11-25。当前尚不能宣称100分钟实验完成，尚未生成该运行的性能图。

## 已执行

- 源码 `148cf54`，服务器195项测试通过；已推送远程。
- 新建隔离工作树 `/root/PaperTest100`，没有覆盖原 `/root/PaperTest` 的未提交文档。
- 从800万行工作集的不可变冷快照恢复，完成恢复审计，再启动 `admtune.timed_static --minutes 100 --until-trial 100`。
- 静态原生 Sysbench Read-only、uniform访问、32线程、15秒预热+60秒测量；仅调数据库参数。100是分钟预算，不是保证执行100次。
- 使用真实 `qwen3.7-plus`，参数选择、范围裁剪、候选推荐、GP预筛、周期增量反思均走真实执行入口；不启用启发式降级。末端复测与恢复预留在预算内。

## 最后成功读取的进度

服务器时钟 `2026-11-25 03:50:35 CST`，进度快照活动耗时1150.23秒：

| 项目 | 已核实值 |
|---|---:|
| 完成试验 / 有效观测 | 14 / 14 |
| 初始配置吞吐 | 2459.08 TPS |
| 接受的搜索最优吞吐 | 3789.88 TPS |
| 搜索最优相对初始提升 | 54.12% |
| 真实API调用 | 4 |
| 实际用量 | 5174 tokens |
| GP选中新候选 | 2 |
| 启动候选 | 4 |
| 抽样审计 | 1 |
| 已有最优配置重复 | 5 |
| 初始配置穿插复测 | 1 |
| 初始基线 | 1 |

搜索值不是末端独立复测值；没有拿重复点当作新候选。
完整的已读检查点摘要见 `docs/experiment_data/timed_static_100min_001/last_verified_status.json`。

## 阻塞与恢复顺序

随后SSH连接中断，有限重试后，18594与之前提供的12972两个端口均拒绝连接。不能据此判断后台进程已停止、已完成，或数据库已恢复；也不能把“最后看到14次”当作最终次数。

服务器恢复后，必须先读以下文件与进程，不可直接重复启动、恢复快照或覆盖实验目录：

1. `/root/autodl-tmp/admtune-deploy/timed-static-100min-001.exit` 与同名前缀 `.log`。
2. `/root/PaperTest100/runs/timed-static-100min-001/summary.json`、`progress.json`、`records.sqlite3`。
3. 若仍运行，只监测并等待；若完成，校验原始Sysbench日志、GP快照、模型用量、最终复测及配置/行数恢复后取回归档。
4. 若中断，先核查未完成attempt和数据库状态；保持原始失败记录，不把未知耗时补成精确时间，不把后续会话拼进首个100分钟。

本地临时目录已准备收集、核验、中文报告生成脚本；尚未对当前运行执行成功，因为完整记录未取回。没有生成伪造曲线。

## 已准备的绘图材料

- `docs/analysis/extract_agenticdb_fig5.py`：从用户提供的AgenticDB PDF第9页图5、MS Sysbench Read子图提取原始矢量折线。PDF固定SHA校验，不能直接套用到别的版本。
- `docs/analysis/plot_timed_static.py`：只接受通过原始日志核验的100分钟运行数据，缺少 `verification.json` 和 `admtune_trials.csv` 会停止，不生成模拟数据。
- 原文基线已保存为 `literature_curves.csv` 与 `literature_provenance.json`。六方法为ADMTune、AgenticDB、AgentTune、GPTuner、CDBTune、Default；横轴为 Tuning Time (min)，主图纵轴为Throughput (TPS)，另输出同协议吞吐/延迟图。
- ADMTune必须来自本次真实记录；原文基线明确存储论文来源，不能称为本服务器实测。原文优化TPS/P95且AgenticDB含OS调参，本次优化TPS且只调DB参数；这些是跨实验参照，不能当作同协议排名。
- 原文AgenticDB曲线保留原始终止时间，不补造100分钟后续点；本次也不把缺失点、恢复预留时间画成额外试验。

当前需要恢复服务器连接并取回完整数据，之后才能画出可信的本次性能图。
