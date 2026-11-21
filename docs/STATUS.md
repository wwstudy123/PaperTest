# ADMTune staged implementation status

Updated 2026-11-21. Implementation status is separate from experimental evidence.

Local verification: 54 tests passed. Static and A-B-A synthetic matrices each
cover seven variants and three seeds. These are not live database experiments.

| Gate | Status | Evidence / next step |
|---|---|---|
| Repository provenance and secret/output ignores | Implemented | Separate commits; ignored local files and generated runs |
| DBMS-only upstream baseline restriction | Implemented | Deterministic action/phase/protected-parameter tests |
| Frozen SQL templates and workload phases | Implemented offline | AST/DBSCAN/unknown-channel/confirmation tests |
| Selector/pruner/multi-candidate roles | Implemented | Heuristic fixture and opt-in inherited API client; strict controller validation |
| Log-target GP and LCB/UCB | Implemented offline | Numerical, noise, encoding and ranking-reversal tests |
| Versioned history and reflection evidence | Implemented | Compatibility, atomic consumption, failure and deduplication tests |
| Synthetic A-B-A and ablations | Implemented | Seven variants, multiple independent simulator seeds; JSON/CSV audit |
| MySQL apply/readback/measure/restore adapter | Fake-connector tested | Not yet verified on the user's server |
| Real static preflight/pilot/tune commands | Implemented, not executed | Follow LIVE_RUNBOOK with local credentials |
| Real dynamic offered-load and replay driver | Pending deployment integration | Must not substitute completion TPS or synthetic windows |
| AgentTune baseline on shared real protocol | Pending | Do not label heuristic proposer or no-GP variant as AgentTune |
| Complete OLAP execution | Pending | No success label for incomplete query sets |
| Formal database results | Not produced | Requires server gates, fixed protocol, repeats and independent verification |

## Cost policy

- Offline tests, simulator and matrix use no LLM API and no database connection.
- Paid role calls require `--backend llm --allow-paid-llm`, configured endpoint/model
  and global call/token budgets.
- No implicit API retries or heuristic fallback in paid mode.
- Failed/unknown-usage calls retain their budget reservation.
- Final verification reserves trial slots rather than running outside the budget.
- Simulation reports count heuristic role invocations separately from API calls.
- Live preflight captures untuned connection/global buffer settings for the joint
  memory screen. This is a screening estimate, not a bound on total process RSS.

## Report interpretation

`suite.csv` and `suite.json` are descriptive plumbing reports, not performance
claims. Target misses remain null instead of being excluded. Prediction errors
and coverage use pretrial predictions on selected online observations, so they
remain selection-biased; a separately sampled audit set is needed for calibrated
generalization claims. Interval width is reported together with coverage.

Per-run manifests freeze seed, catalog, environment IDs, encoding, policy and
source revision. They record installed dependency versions, a dirty-tree flag
and a working-source hash so tests made before commit are not mislabeled as clean.
Live manifests also save initial-state readback and benchmark
protocol. `summary.json` reports final independent values separately from the
best value selected during search. Total live command wall time includes preflight
and final restoration; the tuning ledger separately reports active tuning time.

## Next actual operation

1. On the dedicated Linux MySQL machine, create ignored local configs and `.env`.
2. Run only `admtune.live preflight`; examine version, config values, service path,
   invariant readback, schema and table count.
3. Run manual baseline and a small known-legal delta, without any LLM.
4. Verify recovery, then run a small static heuristic smoke.
5. Only after those gates, select the model and authorize a bounded LLM pilot.
6. Integrate real offered-load telemetry before dynamic experiments; calibrate
   thresholds independently, freeze formal experiment protocols and baselines.

Run one experiment at a time per database instance. The first adapter does not
provide distributed instance locking or automatic data-snapshot restoration.
Do not start concurrent tuning processes against the same service.
