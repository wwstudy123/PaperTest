# Stage 2 offline verification

## Install

Use Python 3.10 or newer in an isolated environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

No test sends API requests or connects to a database. The SQL template catalog
uses SQLGlot AST literal replacement and fixed DBSCAN clusters. Unknown templates
use a separate channel. Calibration must precede evaluation; do not initialize
the catalog or standardizer from future evaluation phases.

## Synthetic smoke experiment

```sh
.venv/bin/python -m admtune.cli --simulate --scenario aba --seed 0 --trials 24
```

Available variants:

- `full`: all implemented mechanisms.
- `single-slice`: aggregate time slices, keeping the same detector.
- `no-memory`: numerical and candidate history is current-phase only.
- `no-surrogate`: deterministic first legal candidate, not AgentTune.
- `no-reflection`: no verbal summaries; numerical records remain.
- `full-reflection`: bounded event-triggered full-history reflection.
- `additive`: additive configuration Matérn + context linear GP.

The proposer is a seeded **heuristic fixture**, not an LLM. All measured values
come from an invented landscape, not MySQL. None may be used as thesis results.
`summary.json` explicitly reports `simulated: true` and zero paid API calls.

Each run creates an ignored directory containing:

- `manifest.json`: seed, environment IDs, knob catalog, encoding, budgets,
  policy, source revision and calibration metadata.
- `records.sqlite3`: append-only measurements, pre-execution prediction events,
  decisions and atomically consumed reflection evidence.
- `trials.jsonl`: portable measurement export.
- `summary.json`: search best, independent verification values, and costs.

## Numerical contract

Fit separate GPs for `log(TPS)`, `-log(complete_latency_s)`, and (only in a
separately designed rate-controlled experiment) `-log(P95_ms)`. Phase baselines
do not relabel history. LCB admission is applied in the log-performance scale.
Posterior latent standard deviation controls admission; observation standard
deviation controls held-out residual/coverage checks. Baseline-estimate
uncertainty is not propagated in this initial implementation.

Joint Matérn 5/2 is the implementation candidate, not a proven best kernel.
Additive mode is retained as an explicit ablation. Active training sets are capped.

## Important limitations before live experiments

- Synthetic thresholds are fixtures, not calibrated real-workload thresholds.
- Offered request telemetry must be independently scheduled. Completed TPS,
  CPU or buffer hit ratio are rejected as phase-detector inputs.
- An initial stage uses its measured current configuration as baseline. Invalid
  baselines stop that stage rather than inventing a label.
- Hard-only startup probes are isolated test-instance probes, not GP-guaranteed
  safe configurations.
- Real data-state restoration, cold/warm-cache protocol and repeated-run isolation
  remain deployment responsibilities and must be recorded before formal testing.
- Real server access and a chosen LLM endpoint are not supplied in this repository.
