# ADMTune implementation and experiment gates

## Provenance

- Upstream: https://github.com/xinyueyangiscas/AgenticDB
- Starting commit: `321cb7ca350bbf382e7a1696ccfff859ac53f315`
- License: upstream MIT license is retained.
- AgentTune supplies the task decomposition, not copied unlicensed source.
- ADWTune, OnlineTune and Reflexion supply research ideas discussed in the thesis.

## Fixed design

1. DBMS-only actions; no OS writes, OS repair, or phase transitions to OS.
2. Programmatic external-workload monitoring, separate from runtime diagnosis.
3. Parameter selection, preferred-range pruning, and multi-candidate generation.
4. Full fixed-space configuration encoding, deterministic validation and deduplication.
5. GP targets `log(TPS)` or `-log(complete latency)`, separately by objective.
   Phase baselines are used for utility and admission, never to relabel history.
6. Pretrial predictions are persisted before observations update the model.
7. Only complete, effective, compatible observations enter the GP.
8. Real execution owns snapshots, readback, warmup, measurement, and recovery.
9. Event-triggered reflection consumes evidence IDs only after validated durable write.
10. Trial, wall-clock and LLM-call/token budgets never reset on workload changes.

## Incremental delivery

| Stage | Gate before committing and pushing |
|---|---|
| 0 Repository hygiene | Ignore credentials, data, outputs, backups; retain provenance |
| 1 DBMS-only substrate | Tests prove OS actions/transitions are rejected |
| 2 Offline ADMTune | Tests for workload state, GP, agents, memory and reflection; deterministic simulator |
| 3 Real execution adapter | Fake-connector tests for apply/readback/rollback; explicit opt-in preflight |
| 4 Experiment protocols | Smoke/synthetic ablations, provenance and cost reports; no fabricated DB results |
| 5 Dedicated server pilot | User supplies environment; validate baseline and recovery before LLM calls |
| 6 Formal experiments | Frozen configs/seeds, independent repetitions and final remeasurement |

Each completed implementation stage is a separate tested commit pushed to the
user repository. A failed push is reported; it is never described as synchronized.
Paid LLM calls and live database mutations require an explicit run command and are
not performed during implementation.

## Stage 1 baseline restriction

`agenticdb --dbms-only ...` restricts the inherited baseline to database actions.
The auditor stops rather than switching to OS tuning. Deterministic proposal and
repair checks reject durability, isolation and instrumentation changes. This
restricted method must be labeled separately from full upstream AgenticDB.

## Evaluation boundaries

- Keep upstream AgenticDB available for provenance; its original entry point is
  not the ADMTune real-run safety boundary.
- Label the restricted baseline `AgenticDB-DBMS-only`, not full AgenticDB.
- AgentTune is a separate method baseline, not the same as a surrogate ablation.
- Synthetic simulator output validates plumbing only; it is not thesis evidence.
- Start with static Sysbench and composition A-B-A. Rate-capped experiments need
  independent offered-load telemetry and a separately defined latency objective.
- Same initial state, knob universe, objective, measurement protocol and budgets
  are required for comparable real experiments.
- Reflection comparisons: none, event/full-history, event/incremental.
- Final configuration performance must be independently remeasured.
