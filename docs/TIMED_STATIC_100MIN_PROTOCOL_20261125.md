# 一百分钟静态性能图实验协议

用户指定一百分钟调优预算，横轴使用分钟，不将其等同于一百次真实试验。

## 本次运行

- 独立工作目录，保留原目录尚未提交的修改，不复用另一容器的第17次检查点。
- MySQL 8.0.36，既有四表各200万行工作集，冷快照恢复后开始。
- 原生 Sysbench Read-only，uniform，32线程，种子42，每次15秒预热、60秒测量。
- 目标为 TPS 最大化，同时保存同配置 Latency；Latency 仍为第95百分位响应时间。
- 参数、持久性、代理核和提示固定；真实 LLM，最多40次尝试及160000 token全局预留。
- 调优时间包含参数试验、模型调用、代理计算和试验后的恢复；部署与冷恢复另记。
- 运行采用100分钟墙钟上限，内部预留最终独立复测及恢复时间；不足以容纳完整试验时提前停止。
- 基准、启动候选、代理选中候选、预测审计、参考、噪声重复、独立复测均分别标记。
- 最高搜索观测与接受配置、最终复测分别保留，重复观测不会被当作新配置收益。
- 本次先生成 Read-only 子图；Write、ReadWrite、TPC-H 的 ADMTune 曲线只能在各自实测后补充。

## 图与来源

六条方法线固定为 ADMTune、AgenticDB、AgentTune、GPTuner、CDBTune、Default。
ADMTune 使用本次实际计时与测量；基线按用户指定的 AgenticDB 论文数据来源整理，
使用其 TPS 分图，而不是将表5的 TPS/P95 分数或 time-to-best 当作 TPS。
图例只显示方法名，来源、数值提取方法与单位在本说明及逐点CSV中保留。
不得由最终表格值线性虚构全过程；只提取实际存在的曲线点。

## 配置输出约束

裁剪智能体只接收已选中参数的目录，且提示明确禁止未选中参数的空值占位；
仍严格检查范围键、硬边界和步长。最多一次格式纠错也消耗同一个真实模型调用预算，
保留失败输出，不静默替换成启发式建议。

## 运行

```bash
python -m admtune.timed_static \
  --target configs/local/workset-v2-target.yaml \
  --benchmark configs/local/timed-readonly-100min.yaml \
  --knobs configs/local/workset-v2-knobs.yaml \
  --restore-audit runs/timed-restore-100min-001/restore.json \
  --output runs/timed-static-100min-001 \
  --minutes 100 --until-trial 100 --dedicated-instance --allow-paid-llm
```

`until-trial` 是额外安全上限；实际次数取决于测量和推理用时。时间限制是停止准入条件，
不是要求占满所有剩余秒数；安全恢复优先。图使用实际完成时刻，不将曲线拉伸到100分钟。
