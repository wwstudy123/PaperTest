# 最小实验交付：动态实测图＋原生事务对照

## 完成状态

事务四方法已于2026-11-28北京时间00:01全部完成，25条有效测量、30次真实API、92812 token，恢复与校验均通过。原件已下载双份保存，详见[事务结果](STATIC_QUICK_RESULTS_20261128.md)。下文服务器段的“正在运行”为启动时历史状态，不是当前进度。

ADMTune最终TPS2473.74、P9517.01 ms、TPS/P95145.43；本轮组合值最高但不代表稳定排名。动态与静态图及CSV已具备，不继续自动追加收费实验。

## 范围与预算

本次按“复用已有数据、最少新增收费实验”执行，不再重跑已完成的四方法动态种子43，不新增100分钟搜索，不引入TPC-H、TPC-DS、JOB或文献比例曲线。

1. **动态图**：从本地完整原件离线整理，不调用LLM或数据库。
2. **事务对照**：在新服务器重建同规模四表、每表200万行，使用原生Sysbench只读事务、32线程、15秒预热＋60秒测量。每方法报告TPS、P95和TPS/P95。
3. 保留既有事务协议的优化目标 **TPS/P95最大化**，不是事后改成TPS最大化或宣称吞吐与延迟同时改善。
4. Default三次测量；各调优方法最多8次测量（含初始与最终独立复测）、20次API、64000 token、1500秒活动预算。未用完不凑数，错误不自动付费重跑。

这是单次最小交付，不等于多种子动态主实验与四项消融已经完成。

## 新服务器状态

2026-11-27晚连接的是新实例`autodl-container-77c34b852e-98bb92e3`，不是原动态首轮实例。新数据和硬件身份独立留档，不能将其结果与旧实例拼成同环境独立重复或连成一条搜索曲线。

- 部署目录：`/root/PaperTestQuick`，执行源码`2d37047`；本次仅补公开参数清单，不修改搜索算法。
- 原800万行工作集已重新生成，独立3308实例、冷备与恢复校验通过。
- 服务器294项回归测试通过。
- 事务批次：`/root/experiment-results/static-quick-20261127`。
- 顺序：Default→AgenticDB（DB-only）→AgentTune→ADMTune；`repeat=1`是运行编号，不是可控LLM种子。
- 结果以`batch.json`及各方法账本为准。本文件建立时批次正在运行，没有提前填写事务性能结果。
- 公共负载和参数清单在`configs/experiments/static-readonly-*.yaml`；私有目标配置与密钥不提交。

## 已交付：四方法动态实测图

数据来自原实例的2026-11-27凌晨种子43首轮，各方法均为30分钟A→B→A、18000条成功请求。没有重新压测，没有平滑、补点或虚构误差带。

| 方法 | 全程P95（ms） | 平均延迟（ms） | A1达标耗时（秒） | API | token |
|---|---:|---:|---:|---:|---:|
| Default | 72.42 | 45.48 | 未达标 | 0 | 0 |
| AgenticDB | 53.83 | 34.34 | 160 | 9 | 68655 |
| AgentTune | 56.15 | 36.81 | 180 | 33 | 52223 |
| ADMTune | 59.50 | 38.05 | 560 | 12 | 15735 |

三种调优方法在B、A2均于首组60秒窗口达标，不等于在60秒内学习出新配置。ADMTune在A2没有执行新候选（GP全拒），但基准、复测与全过程服务均完成。其相对Default降低17.84%的全过程P95，本轮仍慢于两种调优基线；token更少不等于总体性能最优。

- 图：`experiment_data/dynamic_matrix_r0_20261127/figures_and_data/dynamic_performance.png/.pdf`
- 开销：同目录`tuning_overhead.png/.pdf`
- 适应：同目录`adaptation.png/.pdf`；NA表示未达标，不是耗时0。
- 可编辑数据：`overall_metrics.csv`、`window_metrics.csv`、`phase_metrics.csv`、`adaptation_metrics.csv`、`successful_request_curves.csv`。
- 来源：`provenance.json`保留每方法逐请求原件SHA256及共同请求计划哈希；私有原始记录仍在本地忽略目录和原服务器。

复现绘图（输出目录必须未存在）：

```bash
python -m tools.plot_dynamic_matrix \
  --batch runs/dynamic-matrix-r0-20261127/dynamic-matrix-r0-20261127 \
  --output runs/dynamic-figures-rebuilt
```

## 后续决策

事务首轮完成后先交付图和完整性能表，再决定是否追加重复。若论文要主张动态模块有效，仍需动态配对重复、无GP及其他消融；本次小预算事务对照不能替代它们。
