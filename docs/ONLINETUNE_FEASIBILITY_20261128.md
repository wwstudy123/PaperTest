# OnlineTune接入可行性检查

## 结论

**可以将此仓库作为基线适配的来源，但目前不能直接运行，也没有产生可以加入正式折线图的OnlineTune数据。** 本轮仅做源码、语法和资产检查；未安装原依赖、未执行数据库或系统脚本、未调用LLM，未改动现有ADMTune和基线执行逻辑。

- 来源：`https://github.com/DBoptimizer/OnlineTune.git`
- 固定提交：`88ba8c35f554bc4a4f7f1a88257b3e8928f87a39`
- 本地原件：`runs/upstream_sources/OnlineTune-audit/`，在Git忽略目录中。
- 32个Python文件逐一通过AST解析检查，其中2个不能解析。
- 机器可读证据：`experiment_data/onlinetune_audit_20261128/source_audit.json`，含文件SHA256、缺失资产和行号。

“源码有问题”不等于算法不可用；但不能把适配成本说成只改端口和路径。

## 已确认可利用的算法机制

| 机制 | 源码位置 | 状态 |
|---|---|---|
| 配置Matérn-5/2＋上下文线性核 | `onlinetune/safe/models/gp.py` | 实现存在，有关键字拼写等兼容问题 |
| DBSCAN上下文聚类、SVM区域选择 | `onlinetune/dbscan.py`、`onlinetune/tuner_safe.py` | 实现存在，依赖历史数据和原维度假设 |
| 安全置信界、信赖域与线搜索 | `onlinetune/safe/safeopt.py`、`subdomainbo.py` | 实现存在，后者目前语法错误 |
| 数据库反馈和逐次模型更新 | `tuner_safe.py`、`safe/model.py` | 路径存在，尚未通过运行验收 |
| SQL编码器 | `Workload2Vec/LSTM/model/` | 仓库包含checkpoint和模型权重；不是所有模型文件都缺失 |

这些机制应在适配中保留；不能只取一个GP核函数，就把所得方法称为完整OnlineTune。

## 直接运行的阻塞

### 1. 入口和核心搜索文件不能解析

- `scripts/optimize.py:26`：参数定义前多了缩进，实际AST错误为`unexpected indent`。
- `onlinetune/safe/subdomainbo.py:393`：多出的右括号，实际AST错误为`unmatched ')'`。

这两项在加载数据库前即可阻止运行，不是推测出的性能问题。

### 2. 模块、词表和基准预测器不完整

- `dbenv.py:14`导入`onlinetune.utils.dynamic_worklod`，仓库中没有该模块。
- `dbenv.py:87`读取`Workload2Vec/LSTM/model/vocabulary.txt`，该文件不在仓库。
- `safe/real_database.py:31`加载`tune.model`；仓库根目录和README工作目录`scripts/`均没有该文件。它用于根据上下文预测默认性能，不能填随机值冒充。
- README要求安装的`setup.py`不存在。
- 入口导入`SYSBENCH_WORKLOAD`、`TPCC_WORKLOAD`等名称，而当前`workload.py`只定义OLTPBench和JOB两个配置。
- `safe/model.py`、`safe/rule_based.py`仍使用`autotune`旧命名空间。

需要优先寻找完整发布版本或向作者索取缺件。若必须重建词表或基准预测器，要记录训练来源、标签和数据预算；不能使用正式测试后验数据。

### 3. 旧依赖与额外运行问题

`requirements.txt`固定TensorFlow1.14和scikit-learn0.22.2，并依赖GPy、GPyOpt等；不能不经验证就安装进现有Python3.10/3.12环境。应使用单独环境，不升级或降级现有ADMTune实验环境。

源码还存在GP核参数`ctive_dims`拼写、`np.asscalar`旧接口、`dbenv.py`使用但未导入`tf`等问题。上述是源码检查发现，尚未完成逐项运行修复。

### 4. 与当前执行器、参数空间和评价目标不匹配

- 原实现多处假定40个参数和10维上下文，如`dbscan.py:51`和`safe/model.py:57`；当前实验是4或5个共同数据库参数，不能只改JSON清单。
- 原规则直接读取`max_join_size`等完整目录字段，小目录会缺键。需处理“未搜索但保持固定”的参数引用，并确保不能越出共同硬目录。
- 原目录含`query_cache_size`等不能直接套用到本MySQL8.0目录的条目；也不能允许改变共同持久性条件。
- 原执行脚本偏向OLTPBench/JOB，并固定外部路径、64线程及约180秒评测；与当前Sysbench32线程15＋60秒及连续SQL流不同。
- `--y_variable=lat`虽在CLI出现，但可见奖励路径仍取`external_metrics[0]`作为吞吐，不能仅改命令行就宣称在优化P95。
- 原实现自行切换工作负载、启停general log，并在压测后清理连接。必须换成现有隔离执行接口，避免干扰连续请求流或其他实例；不原样运行它的清理脚本。

## 推荐的最小接入路线

### 第一步：建立独立适配验收，不立即付费大跑

1. 冻结源码与修补清单，修正语法、命名空间和API兼容。
2. 确定缺失词表、`gen_context`和`tune.model`的补齐方案，明确数据成本。
3. 先验证“给上下文→选配置→接收真实观测→更新模型”的接口，保留聚类选择、安全模型及信赖域/线搜索。
4. 用现有统一执行器处理参数映射、回读、恢复、记录和真实试验预算；OnlineTune本身不需LLM调用。
5. 输出原始观测、算法状态和补丁清单。删改关键机制时要说明是适配版，不宣称未修改原实现。

这一步未通过，不安排完整矩阵，也不把普通GP或启发式替身命名为OnlineTune。

### 第二步：先选清楚要补哪张图

- **新实例静态TPS/P95图**：如实例、数据、配置和协议未变，可以优先安排相同目标/硬目录/8观测预算的单方法实测，再与该实例已有四方法结果整理。此项只能说明静态适配表现，不能检验OnlineTune完整动态价值；离线历史或额外默认测量也必须记账。
- **动态A→B→A图**：更契合此基线的研究目的，但现有动态四方法首轮来自旧实例，当前服务器是新实例。新实例只跑OnlineTune，不能直接与旧四条曲线合并为同环境正式排名。应在同一实例补齐配对方法；可先做Default、ADMTune、OnlineTune小组检验，再补AgenticDB、AgentTune。

另一个问题是数据预算：当前协议从空历史开始，而原实现使用默认性能预测器和聚类历史。需要统一提供训练/校准条件或显式计入预算，不能让任一方法获得隐含的测试答案。

### 第三步：取得完整实测后再加线

绘图按真实工作负载时间/实际调优时间，保留阶段切换、失败和恢复。不要把8个真实点扩成100次试验；不要对新旧实例做比例缩放后称为统一实测。已有结果与图无需因本次检查修改。

## 公开代码说明

该提交未发现LICENSE/COPYING文件。原件留在忽略目录中；不将整套上游源码复制到公开仓库。后续公开适配器及补丁前应确认授权或采用独立实现并明确引用。

## 当前决策

**值得接入，但属于需要修复与模型资产补齐的适配任务，不是可立即出图的现成基线。** 最省时间的下一步是先解决缺件并跑通独立接口验收，而不是在服务器反复尝试整份旧依赖。
