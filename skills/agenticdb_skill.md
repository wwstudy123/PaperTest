# 数据库参数调优规划提示词

{{task_brief}}

## 输入资料

以下是本轮提供的压测参考文件：

{{benchmark_files}}

先读取 `workload_interpretation`，再与实际执行的脚本、模式和命令核对。不能因为脚本中存在未执行的读写分支，就把只读负载误判为读写负载。

当前调整阶段为 `{{current_phase}}`。仅允许调整数据库参数，没有其他可切换的参数层级。

## 参数与收益规则

1. `allowed_knob_space` 是唯一候选目录。不得虚构参数、扩大硬边界或修改压测输入。
2. `current_config` 与目录中的当前值来自实际回读。候选未改字段由控制器继承。
3. `persistable=false` 的参数只能在线生效，不得依赖写入启动文件或重启后仍生效。
4. 保持持久性、事务隔离、一致性和观测能力不变，不以放松这些语义追求分数。
5. 依据当前负载、资源诊断、兼容历史和最新执行反馈规划。系统指标仅供诊断，不是可修改对象。
6. 可以描述整体数据库配置计划，但本轮执行项必须只包含合法且有依据的参数变更；不强制凑满参数数量。
7. 优化目标以 `objective` 为准。TPS、TPS/P95 和完整查询执行时间不能互相替换。
8. 未完整完成的测量不能视为成功。组合调整只能确认组合效果，不直接推断单参数因果关系。
9. 后续轮次不重复首轮完整解释；保留成功方向，避免重复已核验失败组合。
10. 不输出 shell 命令、内核参数或系统控制项。数据库配置应用、重启、预热和恢复均由控制器执行。

## 后续动作与停止建议

`action_type` 只能是 `db_config` 或 `repeat_benchmark`。重复测量时 `candidate_config` 必须为空。

`auditor_recommendation.next_phase` 只能是 `db` 或 `stop`。数据库搜索达到平台期后建议停止，不建议其他调整层级。是否接受候选、何时停止，由控制器和有效实测决定。

## 输出结构

只返回一个 JSON 对象，字段名保留如下。以下为结构示例，具体参数必须来自候选目录：

```json
{
  "diagnosis": "依据观测的瓶颈判断与本轮依据",
  "action_type": "db_config",
  "candidate_config": {},
  "restart_required": false,
  "expected_effect": "待验证的预期效果，不是实测结论",
  "risk": "适用条件、潜在退化与恢复要求",
  "validation_required": true,
  "global_config_plan": {},
  "exploration_mode": "normal",
  "next_step": "成功后的后续验证",
  "if_failed_next": "失败后的候选或范围修正",
  "more_aggressive_plan": {},
  "auditor_recommendation": {
    "next_phase": "db",
    "reason": "继续或停止的证据",
    "confidence": "medium"
  }
}
```

不得将未经执行的配置标记为成功，也不得把推测写成已验证事实。
