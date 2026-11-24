# 区分度校准与可续接真实试验协议

## 第一阶段：零 LLM 校准

保留上轮无收益结果，不将新分布混入原图。新入口显式传递并记录 Sysbench `rand_type`。
使用原生只读负载，32 线程、四表各 200 万行、固定种子 42、15 秒预热与 60 秒测量。
恢复 v2 冷快照后依次执行：

- special：1 GiB → 2 GiB → 1 GiB；
- uniform：1 GiB → 2 GiB → 1 GiB → 2 GiB → 1 GiB。

共八次真实测量，不调用 LLM，其他配置与持久性不变。
每个 2 GiB 观测与左右 1 GiB 观测的均值成对比较。预先固定启动门槛：

1. 所有观测有效且最终恢复初始配置；
2. uniform 相邻参考观测相对漂移不超过 10%；
3. 两个候选的相对收益均大于 `max(5%, 2×最大参考漂移)`。

未通过时不得直接启动付费五十次实验，需要保留失败原因并重新分析；
不通过修改图、接受门限或删去不利观测来强行通过。

## 第二阶段：连续编号，而非两个独立运行

目标全局试验上限为 100，先执行至第 50 条并停在持久化检查点。
后续可设累计终点 51、52、…、100，不建立新的基准或清空 GP/经验。
真实执行失败、无效和中断均占用原编号，不在同一编号上偷偷重试。

静态均匀负载用于先建立可信数据源，不宣称其已验证动态负载机制。
基准、启动候选、GP 选中候选、预声明预测审计、参考测量、incumbent 重复与检查点复测
分别标记；五十次真实测量不等于五十个独立候选或五十次成功搜索。
GP 无可试候选时不放宽筛选去伪造收益，可进行明确标记的重复观测以估计噪声。
预声明审计只绕过性能预测筛选，不绕过配置类型、资源、持久性、实际异常恢复或 P95 限制。

## 持久化要求

- 执行前以事务预留编号、配置、种类、试验前预测和决策；完成后再写真实结果。
- 持久化全局调用/token 用量、候选池、选择/裁剪结果、随机数状态、最优配置与训练记录。
- 每次续接核对源码、基准定义、环境、数据、参数空间和数据库初始状态。
- 干净检查点恢复初始数据库配置；中断条目不作为训练标签。
- 第 50 次为检查点独立复测，不改写历史搜索最优；第 51 次保留其历史，继续原序列。
- 原始日志、SQLite 账本、JSONL、GP 快照、CSV 和文件校验和均保留。
- 所有阶段串行使用实例锁；源代码部署与性能测量不并行。

全局真实试验上限 100、LLM 尝试上限 40、token/未知用量预留上限 160000；
这些是安全上限，不表示必须花完。候选批量生成、阶段性反思，避免每次测量都调用模型。
先导校准记录不作为正式搜索训练集或正式比较结果。

## 连续序列的固定细节

- 启动至少六条有效观测后拟合 GP；每批最多八个 LLM 候选。
- 初始参考固定于第 10、20、30、40、60、70、80、90 次；第 50、100 次为独立检查点复测。
- 第 5、15、25、35、45、55、65、75、85、95 次为预声明预测审计（有未测候选时执行）。
- 正常搜索按 LCB 性能准入和 UCB 改进选择；没有合格新候选时测量 incumbent，
  用来估计噪声，不把相同配置的波动接受为新配置收益。连续三次无效会停止诊断。
- 所有试验结束后恢复共同初始配置；算法的推荐父配置独立持久化，不依赖数据库停在最优配置。
- 每十条记录统一反思新增证据；历史文字供后续批次检索。静态上下文恒定，不能据此声称动态机制已有效。
- 干净停机后的空闲时间不计为活动调优时间；原始 UTC 会话记录保留真实间隔。
  未知中断为每条缺失结果保守预留 600 秒，不将该预留称为准确耗时。
- 模型/端点/思考开关、执行代码、依赖、硬件配额、配置/数据/参数空间、分布和随机种子被冻结。
  允许更新结果 Markdown、图和 CSV，但不允许在同一序列中偷偷修改算法。密钥可以轮换，不写入账本。

## 新建与续接命令

以服务器仓库根目录运行；下列本地私有配置不提交 Git。

```bash
# 新序列前恢复同一数据快照，仅创建时执行；先确认校准通过。
python -m tools.restore_workset_v2 \
  --audit-dir /root/PaperTest/runs/continuous-restore-001 --dedicated-instance

python -m admtune.continuous \
  --target configs/local/workset-v2-target.yaml \
  --benchmark configs/local/continuous-uniform-001.yaml \
  --knobs configs/local/workset-v2-knobs.yaml \
  --calibration runs/discrimination-001 \
  --restore-audit runs/continuous-restore-001/restore.json \
  --output runs/continuous-uniform-001 --until-trial 50 \
  --dedicated-instance --allow-paid-llm
```

续接时不另建输出目录、不重新生成数据、不清空经验库：

```bash
python -m admtune.continuous \
  --target configs/local/workset-v2-target.yaml \
  --benchmark configs/local/continuous-uniform-001.yaml \
  --knobs configs/local/workset-v2-knobs.yaml \
  --calibration runs/discrimination-001 \
  --output runs/continuous-uniform-001 \
  --resume --until-trial 51 --dedicated-instance --allow-paid-llm
```

将累计终点改成 52 或 100 即只执行尚缺的试验。
移除 `--allow-paid-llm` 并加 `--check-only` 可检查续接，不执行新的压测或 LLM 调用。
数据库配置实例锁覆盖整次调用；中断时有未完成编号会标为未知结果并保留，
只有确认没有遗留 Sysbench 进程才恢复已校验的初始配置，绝不重复该编号。
