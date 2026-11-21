# ADMTune on AgenticDB

ADMTune is being developed incrementally on the AgenticDB execution artifact.
The upstream source and MIT license are retained. **No real database results
or LLM performance claims have been produced by this fork yet.**

The implementation plan and safety boundaries are in
[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md).
Deployment credentials belong in ignored `configs/local/` or environment
variables, never in committed examples. Generated runs and database backups
are ignored by Git.

## Upstream AgenticDB reference

Core implementation for the AgenticDB paper.

## Contents

```text
main.py, loop.py       Main entry point and control loop
auditor.py             Acceptance, rollback, and phase decisions
config.py              Configuration models and loading
connectors/            MySQL, PostgreSQL, and SSH connectors
memory/                Same-task and cross-task experience memory
profiles/              Workload and metric interpretation
tools/                 Execution, metrics, recovery, and reporting
validators/            Configuration, safety, and result validation
skills/                Prompts loaded by the control loop
LICENSE                MIT License
```

The main control flow is in `main.py` and `loop.py`.
Supporting modules implement auditing, candidate selection, database
connections, validation, rollback, metric collection, and experience memory.
The Markdown files under `skills/` are prompts loaded by the code and
are part of the implementation.

Generated experiment outputs are not stored in this source repository. Each
execution creates a separate directory under `runs/` for its reports, CSV data,
logs, prompts, messages, configuration backups, and state snapshots. The
`runs/` directory is ignored by Git.

## Install

Python 3.10 or newer is required.

```bash
pip install -e .
```

## Run

Prepare three environment-specific YAML files:

- `target.yaml`: database and SSH connection settings;
- `benchmark.yaml`: workload command, metrics, and objective;
- `knobs.yaml`: tunable database knobs.

After installation, run a dry-run session with:

```bash
agenticdb --rounds 15 \
  --config /path/to/target.yaml \
  --benchmark-config /path/to/benchmark.yaml \
  --knobs-config /path/to/knobs.yaml \
  --dry-run
```

The equivalent direct Python command is:

```bash
python main.py --rounds 15 \
  --config /path/to/target.yaml \
  --benchmark-config /path/to/benchmark.yaml \
  --knobs-config /path/to/knobs.yaml \
  --dry-run
```

For an environment-connected tuning session, replace `--dry-run` with
`--real-run`. Use `--no-history` and `--no-memory-book` when a clean ablation
run must not reuse earlier experience.

The complete private server and benchmark deployment environment is not
included.

## License

MIT License. See `LICENSE`.
