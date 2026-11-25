# ADMTune 数据库参数调优实验

本项目基于 AgenticDB 的数据库执行代码，逐步实现动态负载分析、多角色候选生成、代理模型调度和增量反思。保留上游 MIT 许可证与来源记录。

**已保存真实静态调优及配置诊断记录，但尚未完成同协议、多方法、多种子正式实验。模拟结果不能作为实测结果。**

**2026-11-26科研图包**：[下载ZIP](docs/exports/admtune_plot_pack_20261126.zip) · [10组图及数据索引](docs/PLOT_PACKAGE_20261126.md)。本地265项测试通过。新GP臂中断已保留并完成独立收尾，两臂均未接受新配置；正式基线、独立重复和真实消融仍需补齐，见[最新结果](docs/COMPOSITE_GP_PAIR_RESULTS_20261126.md)。

2026-11-25最新进度：综合目标第二轮真实闭环验收通过，7次有效测量、5次API/10641 tokens；未找到优于初始配置的综合指标结果。见[第二轮数据与曲线](docs/COMPOSITE_PILOT_002_RESULTS_20261125.md)。[首轮失败证据及修复](docs/COMPOSITE_PILOT_RESULTS_20261125.md)仍保留。

随后完成9次零LLM重复及固定6/3时间切分分析：重新拟合的均值误差约2%，但留出区间仅覆盖1/3，不能声称GP已可靠校准；见[重复测量与区间诊断](docs/UNCERTAINTY_REPEAT_RESULTS_20261125.md)。

已新增默认关闭的[训练噪声与逐批更新版本](docs/GP_TRAINING_NOISE_V1_20261125.md)，本地和服务器252项回归通过。[独立零LLM验收](docs/GP_NOISE_VALIDATION_RESULTS_20261125.md)完成9次实测，两版GP均覆盖3个检验点、误差约0.54%；不代表新版总体更优，旧付费控制器未自动切换。

新GP已作为显式选项接入综合目标控制器，默认仍为legacy；[同预算闭环先导协议](docs/COMPOSITE_GP_PAIR_PROTOCOL_20261125.md)已冻结，校准重复计入预算，不混用旧数据。

## 功能边界

- 仅调整数据库参数，操作系统参数调优代码已删除，不再提供启用开关。
- 保留 CPU、内存、磁盘和 I/O 的只读诊断；这些信息不会形成系统参数修改动作。
- 大语言模型按参数选择、范围裁剪、候选推荐和反思任务调用，不训练或微调模型权重。
- 高斯过程使用实测性能的对数目标；阶段基准用于收益比较，不重新定义历史标签。
- 真实试验只能在专用测试实例上执行；配置应用、实际回读和恢复验证由程序负责。

## 安装与离线检查

需要 Python 3.10 或更新版本：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m admtune.cli --simulate --scenario aba --seed 0 --trials 24
```

上述命令不连接数据库，也不调用模型接口，不需要 GPU。候选由可复现的启发式测试替身生成，性能值来自人工构造的函数，而非 MySQL。

详细说明见[离线验证文档](docs/OFFLINE.md)。

## 真实 MySQL 实验入口

先按[实机运行手册](docs/LIVE_RUNBOOK.md)准备本地配置，再依次进行：

1. `admtune.live preflight`：预检，不应用配置、不重启、不压测、不调用模型。
2. `admtune.live pilot`：不用模型，验证手工小改动、测量和恢复。
3. `admtune.live tune-static`：通过前两步后再开展小预算静态搜索。

真实试验必须显式传入 `--dedicated-instance`。数据库服务重启需要额外的 `--allow-restart`。付费模型调用还需要 `--backend llm --allow-paid-llm` 和明确的调用、令牌预算。

另提供不调用模型、不修改配置的真实外生 A→B→A 回放入口，见[动态回放与绘图记录](docs/DYNAMIC_REPLAY.md)。该入口只验收到达监测，不是动态调优。

真实动态闭环试运行入口为 `python -m admtune.dynamic`，见[动态闭环手册](docs/DYNAMIC_LOOP.md)。采用独立调优线程和同一请求流的 P95，先零 LLM 验收，再显式授权付费角色；不是正式基线或消融结论。完整 OLAP 和同协议论文基线仍待完成。

阶段时长先用 `admtune.phase_safety` 按实际 benchmark 规划；跨阶段观测保护及验收入口见[阶段预算与观测栅栏](docs/PHASE_SAFETY.md)。24 秒监测烟测不能直接套用 75 秒静态测量。

## 合成消融矩阵

```sh
.venv/bin/python -m admtune.experiments --simulate \
  --scenario aba --seeds 0 1 2 --trials 24 \
  --output runs/aba-suite-001
```

运行七个变体、三个随机种子，生成 JSON 和 CSV 审计文件。命令不会隐式启动真实试验或模型调用。`static` 可替换 `aba`，用于静态场景检查。

## 代码目录

```text
admtune/       ADMTune 控制器、负载、代理模型、协作角色、反思与实验入口
connectors/    数据库与 SSH 连接
memory/        从 AgenticDB 继承的会话与经验管理
tools/         数据库应用、恢复、压测、只读诊断与报告
validators/    参数、运行结果和动作权限校验
skills/        运行时读取的中文规划与数据库修复提示词
configs/       可公开的配置示例；私有配置放在 configs/local/
tests/        单元测试、模拟连接器测试及离线集成测试
docs/         设计、状态与运行说明
```

## AgenticDB 裁剪版基线

保留的 `agenticdb` 命令及 `main.py`、`loop.py` 用于数据库层基线。当前分支已删除系统调优分支，达到数据库搜索平台期后直接停止，无需也不再接受 `--dbms-only` 参数。

```sh
.venv/bin/python main.py --rounds 15 \
  --config configs/local/target.yaml \
  --benchmark-config configs/local/benchmark.yaml \
  --knobs-config configs/local/knobs.yaml \
  --dry-run --no-history --no-memory-book
```

这个裁剪版必须标记为 `AgenticDB-DBMS-only`，不能称为完整原版 AgenticDB。继承入口不等同于 ADMTune 实机入口的全部安全检查；实际试验优先按实机手册操作，不直接套用上游的真实运行命令。

## 文件与凭证管理

凭证只写入被忽略的 `.env`，部署配置只放入被忽略的 `configs/local/`。`runs/`、数据、日志、配置备份和虚拟环境均不提交。不要将密码或模型密钥发到聊天、写入公开示例或加入 Git。

- [实施方案](docs/IMPLEMENTATION.md)
- [当前状态与后续门槛](docs/STATUS.md)
- [上游 AgenticDB](https://github.com/xinyueyangiscas/AgenticDB)

## 许可证

保留上游 MIT 许可证，原文见 `LICENSE`。许可证不是 Markdown 文档，不作翻译或改写。
