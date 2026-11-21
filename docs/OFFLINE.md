# 离线验证与合成消融

## 安装与测试

使用 Python 3.10 或更新版本的独立环境：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

自动化测试不发送模型请求，也不连接数据库。SQL 模板模块使用 SQLGlot 对语法树中的字面常量进行替换，再建立固定 DBSCAN 聚类。未知模板进入独立通道。

模板映射和标准化尺度须由独立初始化数据确定，不能使用正式测试未来阶段的信息。

## 最小模拟运行

```sh
.venv/bin/python -m admtune.cli --simulate --scenario aba --seed 0 --trials 24
```

候选生成器是带种子的启发式测试替身，不是大语言模型。性能来自人工函数，不是数据库。`summary.json` 中明确标记 `simulated: true`，模型接口调用数为零。

## 消融变体

| 参数值 | 含义 |
|---|---|
| `full` | 使用已实现的完整离线机制 |
| `single-slice` | 聚合窗口内时间片，保留阶段判断流程 |
| `no-memory` | 数值历史、候选经验与文字摘要只允许使用当前阶段记录 |
| `no-surrogate` | 不训练或使用代理，按确定顺序选择合法候选；不是 AgentTune |
| `no-reflection` | 关闭文字反思，保留数值记录与模型更新 |
| `full-reflection` | 事件触发的有上限全历史反思对照 |
| `additive` | 配置 Matérn 核与上下文线性核相加的对照 |

## 多种子矩阵

```sh
.venv/bin/python -m admtune.experiments --simulate \
  --scenario aba --seeds 0 1 2 --trials 24 \
  --output runs/aba-suite-001
```

每种场景运行七个变体、三个种子，共 21 次。静态场景将 `aba` 改为 `static`，并使用新的输出目录。输出目录已存在时拒绝覆盖。

## 输出文件

所有产物写入被 Git 忽略的运行目录：

- `manifest.json`：环境、种子、参数目录、编码、预算、策略、依赖版本和源码状态。
- `records.sqlite3`：原始测量、试验前预测事件、决策及反思证据消费记录。
- `trials.jsonl`：可移植的测量记录导出。
- `summary.json`：搜索最优值、独立复测值及成本。
- `audit.json`：单次运行的描述性指标。
- `suite.json`、`suite.csv`：多变体、多种子的汇总与失败清单。

## 数值模型定义

吞吐量目标采用 `log(TPS)`，完整查询执行时间采用 `-log(complete_latency_s)`。仅在另行设计的限速实验中，才使用独立的 `-log(P95_ms)` 模型。不同目标不混训。

阶段基准不用于重写历史标签。置信下界在对数性能尺度上判断准入；潜在函数标准差用于准入，观测标准差用于预测残差与覆盖率评价。初版尚未完整传播基准估计误差。

联合 Matérn 5/2 是当前实现候选，不是已证明最优的核。加性核作为消融保留，活跃训练集设置容量上限。

## 尚不能作为实机结论的部分

- 合成阈值是测试夹具，不是经过真实负载校准的阈值。
- 完成 TPS、CPU 利用率和缓存命中率不能替代外部计划请求信号。
- 阶段基准无效时暂停该阶段，不补造性能标签。
- 启动探测只允许在隔离实例中进行，不宣称具有代理模型的性能保证。
- 数据快照恢复、冷热缓存控制和重复运行隔离仍须在正式协议中明确。
- 当前模拟矩阵没有证明任何真实数据库收益或模型成本优势。
