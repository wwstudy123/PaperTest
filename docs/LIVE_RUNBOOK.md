# 专用 MySQL 实例预检与首轮实验

本阶段提供真实执行适配器和显式命令。自动化回归不连接真实数据库、不调用付费 API；服务器上的部署验证单独记录。模拟连接器测试不能代替服务器验证。

## 0 前置检查

- 只允许使用可恢复的专用实验实例，禁止生产库。
- 固定 MySQL 8.0.36、数据规模、数据快照、硬件分配和物理设计。
- 确认实际 MySQL 服务名和启动使用的配置文件。`/etc/my8.cnf` 只是来自
  已有环境的示例，文件存在不等于服务确实读取它。
- 当前实机适配器只验证 Sysbench TPS；P95 是独立约束，不采用 TPS/P95。
- 操作系统参数调优功能已从源码删除，仅保留只读诊断。重启默认禁止，需要 `--allow-restart` 明确授权。
- 当前不会自动重建或还原数据库数据快照。每次独立重复前由操作者还原，
  并把快照编号写入配置。读写负载会改变数据状态。
- 不使用平台会显著降配 CPU/内存的无卡模式运行压测。

## 1 安装与配置

### 容器内本地运行

在数据库所在的专用 Linux 实例执行时，为 `admtune.live` 增加 `--local`，
无需填写 SSH 密码。root 直接读写该实验实例的配置，不依赖 sudo；非 root
仍需明确的文件权限或 sudo 权限。

没有 systemd 的容器可在目标文件的 `mysql` 段设置：

```yaml
service_manager: process
mysqld_path: /实验目录/mysql/bin/mysqld
mysqladmin_path: /实验目录/mysql/bin/mysqladmin
socket_path: /实验目录/mysql.sock
pid_file: /实验目录/mysql.pid
startup_timeout_s: 120
```

这些必须是对应专用实例的绝对路径。预检核对实际 socket 和 PID 路径；
重启前再次核对 PID 的完整启动参数，只关闭该实例，不按进程名批量杀进程。
服务重启仍然需要 `--allow-restart`。这是数据库生命周期管理，不是系统参数调优。

建议数据、MySQL 二进制和大体积产物放在数据盘，代码目录保持独立。
容器中 `free`、`nproc` 可能显示宿主机资源，正式记录应核对 cgroup 配额。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
mkdir -p configs/local
cp configs/examples/target.yaml configs/local/target.yaml
cp configs/examples/benchmark.yaml configs/local/benchmark.yaml
cp configs/examples/knobs.yaml configs/local/knobs.yaml
cp .env.example .env
```

填写本地文件：

- `.env`：SSH、MySQL 和（以后才需要的）模型凭证。
- `target.yaml`：真实服务名、配置文件、分配内存、hardware/data/schema ID。
- `benchmark.yaml`：已验证的脚本路径、表数、行数、线程和测量协议。
- `knobs.yaml`：固定可调全集、硬边界、默认值、步长和生效方式。

参数目录是示例，不是对 MySQL 所有版本合法范围的声明。现有参数值必须
落在该实验的合法空间内，否则预检拒绝开始；不会自动修改基线适配示例。
示例采用 120 秒测量、30 秒预热、32 线程和固定 Sysbench 随机种子。
正式实验前通过独立校准确认这些时长和 P95 阈值。

这三个本地 YAML 和 `.env` 都被 Git 忽略。不要把密码发到聊天或提交仓库。

## 2 只做预检

```sh
.venv/bin/python -m admtune.live preflight \
  --target configs/local/target.yaml \
  --benchmark configs/local/benchmark.yaml \
  --knobs configs/local/knobs.yaml \
  --output runs/preflight-001
```

预检读取版本、当前参数、持久性/隔离/观测不变量、表数、schema 指纹、
Sysbench 版本和服务器标识。不应用参数、不重启、不执行压测、不调用 LLM。
上游 SSH 的 sudo 文件读取可能创建临时读副本，不写数据库或操作系统配置。
服务的完整启动参数、容器资源隔离和数据内容一致性仍需人工核实。

## 3 不用模型的手工首轮试验

先不传 `--delta`，仅验证完整基准测量和结束后的恢复：

```sh
.venv/bin/python -m admtune.live pilot \
  --target configs/local/target.yaml \
  --benchmark configs/local/benchmark.yaml \
  --knobs configs/local/knobs.yaml \
  --dedicated-instance \
  --output runs/pilot-baseline-001
```

然后在 `configs/local/delta.json` 写一项已经人工核验、处于合法边界内的
小变更，再执行相同命令，增加 `--delta configs/local/delta.json`。
确认配置文件快照、实际读回、完整测量和初始配置恢复全部成立。

初版不依靠 LLM 修复故障。恢复失败以非零状态停止，保留备份和日志，
等待人工处理。不能在故障未解决时继续自动试验。

## 4 静态搜索

先用不调用模型的启发式候选生成器验证链路（不是 AgentTune，也不是正式 LLM 结果）：

```sh
.venv/bin/python -m admtune.live tune-static \
  --target configs/local/target.yaml \
  --benchmark configs/local/benchmark.yaml \
  --knobs configs/local/knobs.yaml \
  --dedicated-instance --backend heuristic --trials 12 \
  --output runs/static-smoke-001
```

启发式模式不产生模型接口费，但真实压测仍消耗服务器资源。

基准、启动探测、候选和最终独立复测共享试验槽。失败提交不退还预算。
默认预留 3 个槽作最终复测。实机输出与合成模拟分开标记。
每次命令结束恢复初始配置，不自动部署“最佳配置”。

仅当手工首轮试验已通过、模型和预算已确定后，把 `--backend` 改成 `llm`，
并显式增加 `--allow-paid-llm --llm-calls 16 --llm-tokens 60000`。
未配置密钥、接口地址和模型名时，拒绝启动付费角色。
禁用隐式重试和启发式降级；相同请求缓存命中不再次计费。
令牌预算采用保守预留，接口返回 `usage` 后结算；无法取得 usage 的失败调用
不免费处理。实际美元成本需补充所选模型的计费表，当前不虚构费用。

### 模型接口先做一次小检查

在正式角色调用前，可以单独验证接口和 JSON 返回：

```sh
.venv/bin/python -m admtune.llm_check \
  --env-file .env --output runs/llm-check-001 --allow-paid-llm
```

该命令仅发送一次请求，输出上限为 256 个令牌，无隐式重试，不访问数据库。
用量和结果写入运行目录，不保存密钥。静态调优另存 `llm_usage.json`，
按角色记录调用与用量，不把接口检查计入某个调优运行内部的预算。

显式 `--env-file` 的值优先于同名环境变量。按服务商支持情况设置
`OPENAI_ENABLE_THINKING=false` 可避免在小型结构化任务上消耗推理令牌；
不能仅凭模型名假定所有接口都接受该参数。

## 5 尚未通过的真实实验门槛

- Linux 专用服务器上的实际基准、配置应用、回读与恢复验证。
- 独立、外生的请求计划采集与 A→B→A 固定回放驱动集成。
- 真正的 AgentTune 基线和受限 AgenticDB 的同协议正式对比。
- OLAP 全查询集完整性与超时测量适配器。
- 正式阈值校准、独立重复、数据状态恢复和统计分析。

不要把离线 A→B→A 测试描述为已经完成了真实动态数据库实验。
