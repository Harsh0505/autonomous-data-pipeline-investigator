# Retail Pipeline Guardian — Day 2

Retail Pipeline Guardian detects failures in synthetic multi-supplier retail feeds and investigates
incidents through a bounded, evidence-driven LangGraph workflow. Day 2 adds persistent history,
scoped investigation tools, a configurable model provider, and benchmark evaluation. Autonomous
repair, FastAPI, and the frontend remain intentionally out of scope.

## Architecture

The pipeline ingests supplier data, executes nine quality checks, transforms valid prices to GBP,
and aggregates healthy sales. Every run can be persisted to SQLite as execution metadata, schema,
statistical profiles, validation results, and bounded problematic-record samples. Full datasets are
not copied into the history database.

For benchmark scenarios, observable evidence and hidden truth are separate:

- `IncidentReport` v2 contains only facts measured by the pipeline.
- `BenchmarkGroundTruth` remains on the in-memory benchmark run and is passed only to evaluation
  after an investigation finishes.
- Neither the injected failure type nor injection details are written to incident JSON or SQLite.

The investigation agent uses a LangGraph loop:

```text
reason about current evidence → choose one scoped tool → persist evidence → reason again
                             ↘ conclude with cited evidence when sufficient
```

The model determines the path from the evidence. The graph enforces a tool-call budget, captures
tool failures, rejects evidence references that do not exist, and produces an explicit inconclusive
result when support is insufficient.

## Modules

```text
src/retail_guardian/
  config.py                 Supplier contracts and quality thresholds
  synthetic.py              Reproducible supplier sales generation
  failures.py               Failure injection and hidden benchmark truth
  quality.py                Deterministic and statistical validation
  pipeline.py               Ingestion, validation, transformation, and aggregation
  scenarios.py              Healthy and corrupted benchmark scenarios
  storage.py                Repository protocol and SQLite persistence
  investigation_models.py   Versioned hypotheses, evidence, decisions, and reports
  investigation_tools.py    Eight incident-bound, read-only investigation tools
  providers.py              Deterministic test model and OpenAI provider
  investigation.py          Bounded LangGraph investigation workflow
  evaluation.py             Six-scenario benchmark and validation-rule baseline
  cli.py                    Pipeline, investigation, and benchmark commands
```

## Setup

Python 3.11 or newer is required. The checked-in `uv.lock` provides a reproducible environment:

```powershell
uv sync --extra dev --python 3.12
.\.venv\Scripts\Activate.ps1
```

Alternatively, with a standard Python installation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Generate persisted history

Run a healthy baseline followed by an incident against the same database:

```powershell
retail-guardian run --scenario healthy --rows 500 --seed 42 `
  --as-of 2026-10-20T12:00:00Z --output-dir artifacts

retail-guardian run --scenario currency_mismatch --rows 500 --seed 42 `
  --as-of 2026-10-20T12:00:00Z --output-dir artifacts
```

Both commands use `artifacts/history.db` by default. Incident scenarios return exit code 1 because
the validation gate rejected the feed; this is expected. The incident JSON itself does not disclose
the selected benchmark scenario.

## Run an investigation without credentials

Read the incident ID and run the deterministic evidence-driven test provider:

```powershell
$incident = Get-Content artifacts\currency_mismatch-incident.json | ConvertFrom-Json
retail-guardian investigate --database artifacts\history.db `
  --incident-id $incident.incident_id --provider mock `
  --output artifacts\investigation.json
```

The `mock` provider validates the complete graph, tool, persistence, and report infrastructure. It
is a heuristic test/baseline implementation, not an LLM.

## Run a real model investigation

The real provider uses the OpenAI Responses API with strict structured decisions. Credentials and
the model name come only from environment variables or CLI arguments:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "gpt-5.5"

retail-guardian investigate --database artifacts\history.db `
  --incident-id $incident.incident_id --provider openai `
  --max-tool-calls 8 --output artifacts\investigation-openai.json
```

`OPENAI_MODEL` may be replaced with a model available to your account, or supplied with `--model`.
No real API call is made by the automated tests.

## Run the benchmark

```powershell
retail-guardian benchmark --provider mock --rows 500 --seed 42 `
  --database artifacts\benchmark.db
```

The benchmark evaluates all six injected failures after each investigation completes. It reports
root-cause accuracy, evidence support, tool calls, token usage when supplied by a provider,
completion rate, and a simple validation-rule baseline. Ground truth is never passed into the agent,
its prompt, its tools, or persisted investigation state.

## Tests and quality checks

```powershell
pytest -W error
ruff check .
ruff format --check .
pytest --cov=retail_guardian --cov-report=term-missing
```

The tests cover healthy and corrupted pipeline runs, persistence, observable/hidden separation,
history comparisons, all investigation tools, all six diagnoses, repeated investigations, missing
history, failed tools, unsupported evidence, hard termination budgets, and benchmark aggregation.

## Example investigation trace

For a currency-contract incident, the deterministic test trace is typically:

```text
1. get_validation_failures
   Evidence: supplier_currency failed for five rows; unit_price_z_score also failed.
2. get_data_contract
   Evidence: each supplier has one configured contract currency.
3. conclude
   Diagnosis: currency_mismatch
   Support: both evidence IDs are cited in the conclusion.
```

Another finding selects a different second tool: schema failures compare schemas, duplicates and
missing values inspect bounded records, delayed feeds inspect execution history, and distribution
anomalies compare statistical profiles.

## Current limitations

- The generator and data contracts model one retail domain and three suppliers.
- SQLite is suitable for a demonstrable local MVP, not concurrent production workloads.
- Problematic records are intentionally sampled, so dimension impact is sample-scoped.
- Confidence labels express qualitative uncertainty and are not calibrated probabilities.
- The deterministic provider is useful for tests and comparison but is not a substitute for a real
  model investigation.
- The OpenAI provider is implemented but requires valid account credentials and is not exercised in
  the offline test suite.
