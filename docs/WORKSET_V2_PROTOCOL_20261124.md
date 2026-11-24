# 较大工作集与范围/排序查询的零模型验证

## 数据准备阶段

目标不是直接扩大付费调优矩阵，而是先确认可测的参数效应。原 3307 小数据实例和记录保留，新建专用 MySQL 8.0.36 实例：

- 端口 3308；目录 `/root/autodl-tmp/admtune-mysql-v2/`。
- Sysbench 1.0.20 生成四张表，每表 2000000 行，共 800 万行；种子 42、生成线程 4。实际容量以导入后统计为准，不宣称是之前截图的 5000 万行数据。
- 新的完整工作集库为 `admtune_workset_v2_full`，与旧 `admtune_smoke` 分离；标识包含冷备清单的哈希，旧性能记录不跨数据集复用。
- 固定 `innodb_flush_method=O_DIRECT`、缓冲池实例数 1、初始缓冲池 1 GiB、chunk 128 MiB；持久性保持 flush-at-commit=1、sync-binlog=1、doublewrite=ON。以上是数据库实例条件，不是 OS 调优或候选搜索。
- 新目录、端口及本地配置必须未使用；剩余空间不足时拒绝，不覆盖现有实例或配置。
- 凭证仅写入私有 `.env` 新变量及权限 600 的 Sysbench 配置文件，命令行和版本库不保存密码；原模型配置不改动。

准备命令：

```sh
python -m tools.prepare_workset_v2 --dedicated-instance
```

生成后精确检查行数和主键范围、schema 和固定行样本。随后清洁关闭新实例，对 data 与 binlog 同时做冷备，逐文件 SHA256 校验；将原目录保留到 `retained-pre-restore/`，从冷备重新复制到原路径，逐文件核验后启动并再次检查行数、schema 和样本。

此过程实际演练一次物理恢复，不自动删除保留副本。冷备仅供同版本同实例路径使用，不提供在线热备、跨版本恢复或生产数据库备份保证。剩余空间再次检查，必须能同时保留原目录、冷备和恢复后的活动副本。

输出 `preparation.json`、`progress.json`、私有的 `configs/local/workset-v2-*.yaml`。冷备数据不下载或提交 Git，公开报告只保存哈希与检查结果。

## 初次生成失败的保留与修复

首次仅把表数/规模写进 Sysbench 配置文件，1.0.20 的 Lua 脚本选项未生效，实际只生成一张 10000 行表。精确行数校验报错 `Incomplete generated table` 后停止，未进行性能试验。

修复将 `--tables=4 --table-size=2000000 --threads=4 --rand-seed=42` 显式放入命令行，密码仍仅在私有配置文件。`--resume-failed-lua-options` 只接受该特定失败、相同实例配置及原小表仍为 10000 行的状态。保留原失败 JSON、日志和小库 `admtune_workset_v2`，在新库 `admtune_workset_v2_full` 生成完整数据，不删除或覆盖原表；重复修复和覆盖已有输出会拒绝。

## 冻结的查询筛查

使用确定性的范围扫描及必须保留排序语义的查询，返回聚合结果而非向客户端传输大量数据；必须检查访问范围、返回计数和参数生效，并记录排序行数、归并趟数、缓冲池读等诊断指标。

这些查询属于 Sysbench 数据上的自定义派生负载，不是标准 Sysbench 事务、TPC-H 或 TPC-DS。查询、到达率、预热/测量时长、候选顺序和有效性规则在执行前冻结。新数据准备和查询筛查均不调用 LLM，也不自动将记录写入 GP 或经验库。

- 只读、10 QPS、四客户端线程，每流 84 秒，前 15 秒预热、取 `[15,75)` 的 60 秒 cohort。
- 每次选择某表中连续 32768 行；起点保证不越过 2000000。查询只返回一行聚合结果，避免大结果集传输主导延迟。
- **范围扫描块**：`COUNT(*)、SUM(k)、SUM(OCTET_LENGTH(c))`，读取非覆盖的 c 载荷；只比较缓冲池 1 GiB 与 2 GiB。
- **排序块**：对该范围按 `c,id` 排序，保留前 8192 行再计算 `COUNT(*)、SUM(k)`；SUM(k) 依赖保留哪些行，不能简单消去内部排序。只比较排序缓冲 256 KiB 与 8 MiB。
- 各块固定“基准—候选—基准—候选—基准”，种子分别 610/611，共十次观测。其他参数和数据不变，各块内全请求结果摘要必须相同。
- 行数、返回形状、请求完整性、会话参数、generation 和配置均检查；记录 Sort_rows、Sort_merge_passes、缓冲池读等。观测无效或 P95 超过 1000 ms 即停止，不靠缩小范围或删失败记录凑成功。
- 不在各 trial 间自动恢复快照或清缓存，固定预热且夹入基准；冷备恢复演练只是此前的数据恢复验证。两次候选重复不构成独立种子的正式优劣检验。

```sh
python -m admtune.workset_probe \
  --target configs/local/workset-v2-target.yaml \
  --benchmark configs/local/workset-v2-benchmark.yaml \
  --knobs configs/local/workset-v2-knobs.yaml \
  --dedicated-instance --output runs/autodl-workset-probe-20261124-001
```
