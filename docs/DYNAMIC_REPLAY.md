# 真实动态回放与绘图记录

## 本阶段的边界

`admtune.replay` 使用固定的外生到达时间表，对专用 MySQL 8.0.36 的 Sysbench 表执行真实 SQL。它不调用模型、不改配置、不重启数据库、不进行动态参数搜索。UPDATE 会修改数据，不能用于生产库，也不自动还原数据快照。

此处每个请求是一条自动提交的点查或更新。**QPS 不等于 Sysbench OLTP 事务 TPS**，不能将本阶段吞吐和先前静态调优的 TPS 直接比较。

默认每阶段 24 秒，共 72 秒；4 秒完整窗口，每窗两个 2 秒时间片：

| 场景 | A（0–24 秒） | B（24–48 秒） | A（48–72 秒） |
|---|---|---|---|
| 查询组成 | 10 QPS，90% 读 | 10 QPS，10% 读 | 10 QPS，90% 读 |
| 请求强度 | 10 QPS，50% 读 | 30 QPS，50% 读 | 10 QPS，50% 读 |

计划由固定种子 42 提前生成，表号和行号有界。重现的是相同负载组成和强度，不宣称还原了数据库内容或缓存状态，也不要求 A 两段逐请求行号完全相同。

SQL 模板目录来自预声明的两类操作和四张表，不扫描测试阶段来发现模板。语法树模板聚为两个簇并保留未知通道。归一化使用预声明的中性 10 QPS、读写各半窗口，不使用测试观测拟合。变化阈值 3、稳定阈值 1，分别连续两窗确认。本轮阈值只供低速联调，不能冒充独立实测校准。

阶段检测只读取预定到达计数，不读取查询结果、完成速率或 A/B 标签。正常情况下初始阶段在第 8 秒就绪，切换分别在第 32、56 秒确认，新阶段分别在第 40、64 秒就绪。这些是完整窗口与确认规则产生的预期时刻，不是算法优于其他方法的证据。

## 执行

先安装可选依赖，不能在已有压测进行时安装或修改源码：

```sh
python -m pip install -e '.[replay]'
python -m unittest discover -s tests -v
python -m admtune.replay \
  --target configs/local/autodl-smoke-target.yaml \
  --scenario composition --dedicated-instance \
  --output runs/autodl-replay-composition-001
python -m admtune.replay \
  --target configs/local/autodl-smoke-target.yaml \
  --scenario intensity --dedicated-instance \
  --output runs/autodl-replay-intensity-001
```

程序仅在数据库所在服务器本地运行。默认验证四表各 100000 行及连续 ID、访问列、数据库版本；配置文件哈希和关键变量在前后各读取一次。不自动建表或创建数据，也不读取模型密钥用于请求。

## 到达、过载和计时

- 请求按单调时钟的预定偏移发出，不等待前一个请求完成。
- 四个执行线程、最多额外排队 32 个请求。过载时明确记为 dropped，不推迟后续计划、不隐式补发或重试。
- 提交、执行开始和执行结束分别记时；连接建立开销计入对应请求服务耗时。
- 超时设置为连接、读取、写入各 5 秒，失败不是成功性能标签。
- 超过 0.5 秒发出滞后、有丢弃或查询失败，完整性门槛失败。到达计数仍保留，不能删除失败后只画成功曲线。
- offered 时间窗按预定到达归属；实际 emitted 和 completed 时间窗另算。可能存在末窗之后完成的请求，逐请求文件保留其完整时间。

## 必须保留的证据

每个运行目录独立保存，重跑使用新目录，禁止覆盖原始结果：

| 文件 | 用途 |
|---|---|
| `manifest.json` | UTC 起点、源码提交、脏状态、源码哈希、依赖、协议、初始数据库状态 |
| `schedule.json` | 执行前冻结的全部到达计划 |
| `monitor.json` | 模板到簇映射、未知通道、校准特征与固定阈值 |
| `offered.jsonl` | 每次预定到达、实际发出、是否提交和发出滞后 |
| `requests.jsonl` | 查询开始、完成、成功或错误类型；不保存密码 |
| `windows.jsonl` | 每窗多时间片簇计数、检测状态、事件和实际观察时刻 |
| `summary.json` | 完整性、请求数量、阶段就绪时刻、前后配置一致性；明确无调优 |
| `checksums.json` | 上述 JSON/JSONL 和 CSV 的 SHA-256，便于转移后核验 |

### 可直接画图的 CSV

- `requests.csv`：逐请求 scheduled/emitted/started/completed 秒数，operation、表与行、是否提交、成功、错误类型，以及发出滞后、排队、服务、端到端和计划到完成毫秒数。
- `windows.csv`：每窗 offered QPS、实际发出数、按真实完成时刻计算的 completed QPS、丢弃/失败、读占比、该到达窗请求的端到端 P50/P95，以及阶段检测状态。
- `template_rates.csv`：每个 2 秒时间片、每个冻结模板簇的请求数和到达率，适合多时间片图。
- `events.csv`：变化怀疑、变化确认和阶段就绪的窗口终点及实际观察时间，适合叠加竖线。

`cohort_e2e_p95_ms` 是“在该窗计划到达且最终成功的请求”的端到端 P95，不是该窗完成请求的 P95，必须与失败和丢弃曲线一起报告。分位数使用 NumPy 默认线性插值。

CSV 自动生成，也可以离线重新导出，不连接数据库、不调用模型：

```sh
python -m admtune.replay_report runs/autodl-replay-composition-001
```

原始目录和归档保存在 Git 忽略的 `runs/`，不要把 `.env`、私有目标文件和数据库备份打进结果包。公开仓库仅提交脱敏汇总。后续正式实验还要增加动态调优执行、统一数据快照恢复、独立重复、参数空间冻结、竞争方法和消融。

### 既有静态实验的绘图表

静态运行可离线导出，不会再次压测或请求模型：

```sh
python -m admtune.static_report runs/autodl-llm-smoke-002
```

生成 `static_trials.csv`（试验序号、角色阶段、目标值、P95、配置、生效与接受状态、试验前对数预测）、`static_llm_usage.csv`（接口已返回的逐次用量）及 `static_verification.csv`（最终三次复测）。不能把复测回填搜索曲线，也不能将未知用量的失败请求误算为免费。
