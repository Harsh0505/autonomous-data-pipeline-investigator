# Retail Pipeline Guardian — Autonomous investigation and safe repair

Retail Pipeline Guardian detects failures in synthetic multi-supplier retail feeds and investigates
incidents through bounded, evidence-driven LangGraph workflows. Version 0.5.0 investigates
competing upstream causal mechanisms, proposes typed repairs, simulates them against immutable
synthetic snapshots, and verifies the results deterministically. It supports local inference through
Ollama while preserving deterministic providers for reproducible tests. Production mutation,
FastAPI, and the frontend remain out of scope.

The project can run real autonomous investigations locally without API credentials or paid services.

## Architecture

The pipeline persists execution metadata, schemas, statistical profiles, validation results,
bounded problematic-record samples, and synthetic operational records in SQLite. Operational
records cover supplier submissions, execution logs, orchestration attempts, transformation lineage,
and configuration changes. SQLite also stores an insert-once synthetic dataset snapshot for isolated
repair simulation. Full records are not sent to the model. Benchmark labels remain in memory and
never enter incident JSON, SQLite, model prompts, or investigation tools.

The LangGraph workflow repeatedly asks the configured model to make one structured decision:

```text
observable incident + current evidence
                 |
                 v
       choose a scoped tool ------> validate request ------> execute and persist evidence
                 |                                             |
                 +---------------------------------------------+
                 |
                 +------> conclude with real evidence IDs
```

Every provider returns the same typed `ModelDecision`. Ollama receives a state-specific JSON Schema,
uses deterministic generation settings, and has bounded retries for malformed output, connection
failures, and timeouts. The schema is supplied through Ollama's `format` parameter rather than being
duplicated in the prompt. Successful immutable tools are removed from subsequent schemas. Unknown
tools, invalid arguments, repeated tool requests, fabricated evidence references, and unsupported
diagnosis codes are rejected before they can be treated as successful.

For causal investigations, a separate evidence-sufficiency verifier checks the facts cited for the
model-selected diagnosis. It does not choose or insert a diagnosis. Each supported mechanism has a
testable causal signature, such as two matching accepted submissions for a replay, durable writes
on both a failed attempt and its retry, or timely receipt followed by substantial scheduler queueing.
Generic validation failures and unrelated operational records cannot confirm a causal hypothesis.
The verifier reports supporting, contradicting, relevant-but-nondiscriminating, and unavailable
evidence separately.

Root-cause cases use an `identify_underlying_root_cause` objective. Observable validation failures
are treated as symptoms, and their diagnosis codes are excluded from the allowed final schema. The
model must distinguish mechanisms such as two accepted supplier submissions, a retry after a durable
write, and row multiplication in a join. The deliberately incomplete case can only be completed as
`insufficient_evidence`.

Each model turn includes one compact canonical evidence registry, exact citable evidence IDs, valid
failed-check IDs, and an investigation memory. The memory keeps causal questions separate from
their evidence requests, records normalized tool calls, and exposes hypothesis-specific supporting
and contradicting evidence plus the information still needed. One causal question may use several
distinct tools; only equivalent tool name-and-argument requests are duplicates. A compact capability
catalog explains what each remaining tool can answer and what it cannot establish. This keeps
planning structured without introducing a long free-text plan.

The workflow owns hypothesis state while the model emits explicit create, update, confirm, or reject
operations. Stable hypothesis IDs and cumulative evidence survive across turns. Tentative
hypotheses may exist without proof, but confirmation and a diagnosed conclusion must pass factual
sufficiency checks. Counterevidence is shown separately and cannot be cited as support. When no
evidence exists, the causal schema permits only a tool decision while investigation is still
possible; it cannot emit a diagnosed terminal decision.
If a diagnosed claim fails factual validation while operational tools remain, the bounded retry is
constrained to an investigation action instead of allowing the same unsupported conclusion again.
Hypothesis operations are structurally disabled for that recovery decision, so the current state is
preserved until the selected tool returns. Exact and semantically equivalent immutable requests are
rejected before execution. Informative, empty, invalid, and duplicate requests have independent
limits, plus a total-request safety bound. A causal investigation may return insufficient evidence
only after the available operational sources have been inspected or established as unavailable.
Diagnosis taxonomy order varies deterministically by incident so a fixed first enum cannot become a
cross-case default.

SQLite stores each structured model decision, tool name and arguments, returned evidence, updated
hypotheses, and final report. Private model reasoning is neither requested nor persisted.
Rejected structured decisions are recorded separately in the trace with the validation error and a
bounded, schema-only diagnostic payload. Malformed free text and private reasoning are not retained.

## Project modules

```text
src/retail_guardian/
  config.py                 Supplier contracts and quality thresholds
  synthetic.py              Reproducible supplier sales generation
  failures.py               Failure injection and hidden benchmark truth
  quality.py                Deterministic and statistical validation
  pipeline.py               Ingestion, validation, transformation, and aggregation
  scenarios.py              Healthy and corrupted benchmark scenarios
  causal_scenarios.py       Opaque causal cases and neutral operational records
  storage.py                Repository protocol, SQLite history, and complete traces
  investigation_models.py   Hypotheses, evidence, telemetry, decisions, and reports
  hypothesis_state.py       Stable hypothesis operations and transition validation
  causal_evidence.py        Fact-level validation of model-selected causal claims
  investigation_tools.py    Data-quality and operational investigation tools
  providers.py              Mock, Ollama, and OpenAI providers
  investigation.py          Bounded LangGraph investigation workflow
  repair_models.py          Typed proposals, lifecycle state, and verification reports
  repair_operations.py      Scoped deterministic repair operations
  repair_providers.py       Separate mock and Ollama repair-planning agents
  repair_verification.py    Deterministic repair acceptance criteria
  repair.py                 LangGraph planning and isolated simulation workflows
  evaluation.py             Six-scenario evaluation and validation-rule baseline
  root_cause_evaluation.py  Separate ambiguous causal benchmark and metrics
  cli.py                    Pipeline, investigation, trace, and benchmark commands
```

## Python setup

Python 3.12 or newer is required. The checked-in `uv.lock` provides a reproducible environment:

```powershell
uv sync --extra dev --python 3.12
.\.venv\Scripts\Activate.ps1
```

Alternatively:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Install Ollama on Windows 11

1. Download and run `OllamaSetup.exe` from the
   [official Ollama Windows download](https://ollama.com/download/windows). WSL is not required.
2. Open a new PowerShell terminal and verify the installation:

   ```powershell
   ollama --version
   ```

3. Download the initial 4B model:

   ```powershell
   ollama pull qwen3:4b
   ollama list
   ```

4. The Windows application normally runs the server in the background. Verify the local API:

   ```powershell
   Invoke-RestMethod http://localhost:11434/api/tags
   ```

   If the tray application is not running, start it from the Windows Start menu. The standalone CLI
   can instead run `ollama serve` in a dedicated terminal.

The endpoint defaults to `http://localhost:11434` and the model defaults to `qwen3:4b`. Override them
with `--endpoint` and `--model`, or the `OLLAMA_ENDPOINT` and `OLLAMA_MODEL` environment variables.

## Create an incident with historical context

Run a healthy baseline followed by an incident against the same database:

```powershell
retail-guardian run --scenario healthy --rows 500 --seed 42 `
  --as-of 2026-10-20T12:00:00Z --output-dir artifacts

retail-guardian run --scenario currency_mismatch --rows 500 --seed 42 `
  --as-of 2026-10-20T12:00:00Z --output-dir artifacts
```

Incident scenarios return exit code 1 because the validation gate rejected the feed. This is
expected. The observable report does not contain the benchmark scenario or injection metadata.

## Run a real local investigation

```powershell
$incident = Get-Content artifacts\currency_mismatch-incident.json | ConvertFrom-Json

retail-guardian investigate `
  --database artifacts\history.db `
  --incident-id $incident.incident_id `
  --provider ollama `
  --model qwen3:4b `
  --endpoint http://localhost:11434 `
  --timeout 120 `
  --model-retries 2 `
  --max-tool-calls 8 `
  --max-empty-tool-calls 2 `
  --max-invalid-tool-calls 3 `
  --max-duplicate-tool-calls 2 `
  --output artifacts\investigation-ollama.json
```

`ollama` is the default investigation provider, so `--provider ollama` is optional. An unavailable
server, missing model, timeout, or exhausted validation retries produces a persisted failed report
with an explicit cause instead of an uncontrolled loop or fabricated diagnosis.

For credential-free infrastructure tests without a real model, use `--provider mock`. Mock results
are explicitly labelled `deterministic_mock` and must not be interpreted as LLM performance.

## Run an upstream root-cause investigation

Generate one opaque case. Case IDs do not encode their hidden causal label:

```powershell
retail-guardian run-root-cause `
  --case case-001 `
  --rows 500 `
  --seed 42 `
  --as-of 2026-10-20T12:00:00Z `
  --database artifacts\root-cause.db `
  --output-dir artifacts\root-cause

$incident = Get-Content artifacts\root-cause\case-001-incident.json | ConvertFrom-Json

retail-guardian investigate `
  --database artifacts\root-cause.db `
  --incident-id $incident.incident_id `
  --provider ollama `
  --model qwen3:4b `
  --timeout 180 `
  --max-tool-calls 8 `
  --max-duplicate-tool-calls 2 `
  --output artifacts\root-cause\investigation.json
```

The causal fixtures include supplier replays, non-idempotent job retries, incorrect join
cardinality, source omissions, transformation conversion defects, supplier taxonomy errors,
approved taxonomy updates with stale runtime configuration, late supplier delivery, scheduler
backlog, and one case whose audit evidence is deliberately insufficient. Each case also contains
irrelevant but realistic operational records.

## Run the six-scenario regression benchmark

```powershell
retail-guardian benchmark `
  --provider ollama `
  --model qwen3:4b `
  --endpoint http://localhost:11434 `
  --timeout 180 `
  --model-retries 2 `
  --max-tool-calls 8 `
  --max-empty-tool-calls 2 `
  --max-invalid-tool-calls 3 `
  --max-duplicate-tool-calls 2 `
  --rows 500 `
  --seed 42 `
  --database artifacts\benchmark-ollama.db `
  --output artifacts\evaluation-ollama.json
```

The output identifies itself as `real_local_llm` and records per-case and aggregate:

- observable incident-class accuracy and evidence support;
- workflow completion separately from successful resolution;
- informative, empty, duplicate, invalid, and total tool calls;
- tokens, latency, first-attempt schema validity, and retry recovery;
- structured-output validation failures;
- investigation failures and timeouts;
- the simple non-agent validation baseline.

This six-scenario suite is a deterministic incident-class regression benchmark. Its labels can be
derived directly from validation identifiers, so it does not measure ambiguous upstream root-cause
reasoning. Reports include this limitation explicitly.

Run the deterministic comparison separately:

```powershell
retail-guardian benchmark --provider mock --rows 500 --seed 42 `
  --database artifacts\benchmark-mock.db `
  --output artifacts\evaluation-mock.json
```

## Run the root-cause benchmark

The separate ten-case benchmark genuinely requires operational investigation:

```powershell
retail-guardian root-cause-benchmark `
  --provider ollama `
  --model qwen3:4b `
  --endpoint http://localhost:11434 `
  --timeout 180 `
  --model-retries 2 `
  --max-tool-calls 8 `
  --max-empty-tool-calls 2 `
  --max-invalid-tool-calls 3 `
  --max-duplicate-tool-calls 2 `
  --rows 500 `
  --seed 42 `
  --database artifacts\root-cause-benchmark.db `
  --output artifacts\root-cause-evaluation.json
```

It reports root-cause accuracy on resolvable cases, factual evidence support for proposed causes,
correct inconclusive decisions, false confident conclusions, workflow completion,
informative/empty/invalid/duplicate tool calls, tokens, and latency. Evidence support is calculated
from the cited records and their causal relationship to the proposed diagnosis, not from expected
tool names. The deliberately incomplete case is correct only after the agent establishes that the
relevant retry and lineage audit data are unavailable. The deterministic mock deliberately returns
inconclusive for causal cases; it is useful for infrastructure checks, not as a reasoning baseline.

## Safe repair planning and verification

The repair workflow keeps agent responsibilities separate:

```text
completed investigation
        |
        v
repair-planning agent --> validated typed proposal --> explicit simulation approval
                                                        |
                                                        v
immutable snapshot --> isolated repair operation --> deterministic verification
                                                        |
                                                        v
                                              recommended or rejected
```

The LLM may select only a listed capability and validated candidate parameters. It cannot submit
code, SQL, shell commands, or external mutations. Simulation always works on a deep copy of the
insert-once synthetic snapshot. There is no production-apply command; rollback means discarding the
isolated copy.

Supported operations are:

- `deduplicate_records`: remove additional rows only for explicitly selected duplicate order IDs;
- `correct_supplier_currency`: restore one supplier's currency code from its configured contract;
- `activate_approved_category`: load a category only when an approved configuration-change record
  names it.

Verification reruns all data-quality checks and requires every original failure listed in the
proposal to pass. It also rejects new failures, unexpected row loss, changes outside the selected
records or columns, and any mutation of the original snapshot. Business row, order, quantity, and
local-revenue aggregates are recorded before and after. The LLM does not decide the result.

Run a complete deterministic example:

```powershell
$database = "artifacts\repair-demo.db"

retail-guardian run `
  --scenario duplicate_records `
  --rows 500 `
  --seed 42 `
  --as-of 2026-10-20T12:00:00Z `
  --database $database `
  --output-dir artifacts\repair-demo

$incident = Get-Content artifacts\repair-demo\duplicate_records-incident.json |
  ConvertFrom-Json
$investigation = retail-guardian investigate `
  --provider mock `
  --database $database `
  --incident-id $incident.incident_id | ConvertFrom-Json

$repair = retail-guardian propose-repair `
  --provider mock `
  --database $database `
  --investigation-id $investigation.investigation_id | ConvertFrom-Json

retail-guardian show-repair --database $database --repair-id $repair.repair_id
retail-guardian approve-repair --database $database --repair-id $repair.repair_id
retail-guardian simulate-repair --database $database --repair-id $repair.repair_id
retail-guardian repair-result --database $database --repair-id $repair.repair_id
```

Use `--provider ollama --model qwen3:4b` on `propose-repair` to test the real local repair
planner. Investigation-provider and repair-provider choices are independent.

## Inspect reports and complete traces

The investigation output uses schema version 3.1 and contains the conclusion, final hypothesis
state, compact investigation questions and statuses, typed evidence results, provider/model names,
duration, validation failures, tool-result counts, first-attempt schema validity, retry recovery,
and termination status. Version 3.1 adds `investigation_questions` without removing version 3.0
fields:

```powershell
$result = Get-Content artifacts\investigation-ollama.json | ConvertFrom-Json
$result.conclusion
$result.evidence
$result.investigation_questions
```

Read the persisted decision-by-decision trace:

```powershell
retail-guardian trace `
  --database artifacts\history.db `
  --investigation-id $result.investigation_id
```

The trace includes structured hypothesis operations, approved tool arguments, typed evidence, and
the final report. It deliberately excludes private reasoning tokens.
Ollama token totals include every response carrying usage metadata, including structured decisions
that exhaust their retry budget. A recorded zero is kept distinct from unavailable usage metadata.

Version 3.0 intentionally replaced model-authored full hypothesis snapshots with explicit
hypothesis operations and added `result_kind` to evidence. Version 3.1 adds the persisted
investigation-question ledger while retaining all version 3.0 fields. Existing SQLite databases are
migrated on open. Legacy custom providers that still return complete hypothesis snapshots remain
supported and their entries are merged by hypothesis ID rather than replacing workflow state.

## Local-model limitations

`qwen3:4b` is useful for a local demonstration but is not a reliability boundary. It can anchor on
its first hypothesis, choose a tool whose capability does not match its stated question, or require
structured-output retries. Results can vary across Ollama and model builds even with deterministic
sampling. The workflow therefore validates evidence and hypothesis transitions independently,
bounds every request class, and reports failure or insufficient evidence rather than accepting an
unsupported diagnosis. Use the deterministic suites for workflow regression and the root-cause
benchmark for measured model evaluation; neither a single successful run nor the mock provider is
evidence of production-grade causal reasoning.

## RX 6600 XT and performance troubleshooting

The Radeon RX 6600 XT is not listed in Ollama's current Windows ROCm support table. Ollama's Windows
documentation notes that additional Radeon support may use Vulkan, including fallback paths for some
RX 6000 systems. GPU acceleration therefore cannot be assumed on this card; inference may use Vulkan,
partially offload, or fall back to CPU depending on the installed AMD drivers and Ollama build.

If inference is slow or times out:

- update the AMD Radeon driver and Ollama before testing acceleration;
- inspect `%LOCALAPPDATA%\Ollama\server.log` for the selected compute backend;
- close other GPU-heavy applications to free VRAM;
- increase the CLI limit, for example `--timeout 300`;
- keep `qwen3:4b` as the initial model rather than moving to a larger model;
- reduce `--max-tool-calls` only when a shorter investigation is acceptable;
- verify that the server and model are available with `/api/tags` and `ollama list`.

CPU fallback remains functional with 32 GB system RAM, but it can be substantially slower. Benchmark
duration and timeout metrics make that visible rather than treating slow or failed cases as correct.

## Optional OpenAI provider

The existing provider remains available and reads credentials only from environment variables:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "gpt-5.5"
retail-guardian investigate --provider openai `
  --database artifacts\history.db --incident-id $incident.incident_id
```

No paid provider is required for Ollama or mock execution.

## Tests and quality checks

The tests do not require Ollama to be installed or running. HTTP behavior is simulated with
provider-level transports.

```powershell
pytest -W error
ruff check .
ruff format --check .
pytest --cov=retail_guardian --cov-report=term-missing
```

Coverage includes successful structured decisions, dynamic evidence and check-ID schemas, stable
hypothesis transitions, malformed output, actionable retries, fabricated evidence, invalid and empty
tool results, connection failures, timeouts, missing models, repeated-tool prevention, independent
budget termination, inconclusive exits, operational evidence provenance, competing causal scenarios,
fact-level sufficiency for every supported causal mechanism, scenario consistency, multi-step
hypothesis revision, trace persistence, both benchmark suites, and hidden-label isolation.

## Current scope and limitations

- The generator and contracts model one retail domain and three suppliers.
- SQLite is suitable for this local MVP, not high-concurrency production workloads.
- Problematic records are intentionally sampled, so dimension impact is sample-scoped.
- Operational evidence is synthetic and intentionally compact; it does not model every production
  logging system or partial-observability failure.
- Causal sufficiency rules cover the nine benchmark mechanisms. Adding a new supported mechanism
  requires an explicit, reviewed factual contract in the verifier.
- Confidence labels express qualitative uncertainty, not calibrated probabilities.
- Small local models can remain inconsistent despite structured schemas and retries; inconclusive or
  failed reports are valid outcomes.
- Repair simulation supports three reviewed operations and local synthetic snapshots only. Missing
  source values, external systems, scheduler changes, deployment, and production mutation are not
  supported.
- A recommended data repair resolves the verified incident state but does not by itself correct an
  upstream supplier, retry, join, or deployment mechanism.
- RX 6600 XT acceleration is environment-dependent and must be verified locally.
