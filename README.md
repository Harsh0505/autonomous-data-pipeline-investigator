# Retail Pipeline Guardian — Day 2.5

Retail Pipeline Guardian detects failures in synthetic multi-supplier retail feeds and investigates
incidents through a bounded, evidence-driven LangGraph workflow. Day 2.5 adds real local inference
through Ollama while preserving the deterministic test provider and optional OpenAI provider.
Autonomous repair, FastAPI, and the frontend remain out of scope.

The project can run real autonomous investigations locally without API credentials or paid services.

## Architecture

The pipeline persists execution metadata, schemas, statistical profiles, validation results, and
bounded problematic-record samples in SQLite. Full datasets are not copied into the history store.
Benchmark labels remain on an in-memory `BenchmarkGroundTruth` object and never enter incident JSON,
SQLite evidence, model prompts, or investigation tools.

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

Every provider returns the same typed `ModelDecision`. Ollama receives its JSON Schema directly,
uses deterministic generation settings, and has bounded retries for malformed output, connection
failures, and timeouts. Unknown tools, invalid arguments, repeated tool requests, fabricated evidence
references, and unsupported diagnosis codes are rejected before they can be treated as successful.

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
  storage.py                Repository protocol, SQLite history, and complete traces
  investigation_models.py   Hypotheses, evidence, telemetry, decisions, and reports
  investigation_tools.py    Eight incident-bound, read-only investigation tools
  providers.py              Mock, Ollama, and OpenAI providers
  investigation.py          Bounded LangGraph investigation workflow
  evaluation.py             Six-scenario evaluation and validation-rule baseline
  cli.py                    Pipeline, investigation, trace, and benchmark commands
```

## Python setup

Python 3.11 or newer is required. The checked-in `uv.lock` provides a reproducible environment:

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
  --output artifacts\investigation-ollama.json
```

`ollama` is the default investigation provider, so `--provider ollama` is optional. An unavailable
server, missing model, timeout, or exhausted validation retries produces a persisted failed report
with an explicit cause instead of an uncontrolled loop or fabricated diagnosis.

For credential-free infrastructure tests without a real model, use `--provider mock`. Mock results
are explicitly labelled `deterministic_mock` and must not be interpreted as LLM performance.

## Run the six-scenario Ollama benchmark

```powershell
retail-guardian benchmark `
  --provider ollama `
  --model qwen3:4b `
  --endpoint http://localhost:11434 `
  --timeout 180 `
  --model-retries 2 `
  --max-tool-calls 8 `
  --rows 500 `
  --seed 42 `
  --database artifacts\benchmark-ollama.db `
  --output artifacts\evaluation-ollama.json
```

The output identifies itself as `real_local_llm` and records per-case and aggregate:

- root-cause accuracy and evidence support;
- completion status and total duration;
- tool and token counts;
- invalid tool calls and structured-output validation failures;
- investigation failures and timeouts;
- the simple non-agent validation baseline.

Run the deterministic comparison separately:

```powershell
retail-guardian benchmark --provider mock --rows 500 --seed 42 `
  --database artifacts\benchmark-mock.db `
  --output artifacts\evaluation-mock.json
```

## Inspect reports and complete traces

The investigation output contains the conclusion, hypotheses, cited evidence, provider/model names,
duration, validation failures, invalid calls, and termination status:

```powershell
$result = Get-Content artifacts\investigation-ollama.json | ConvertFrom-Json
$result.conclusion
$result.evidence
```

Read the persisted decision-by-decision trace:

```powershell
retail-guardian trace `
  --database artifacts\history.db `
  --investigation-id $result.investigation_id
```

The trace includes structured decisions and hypothesis updates, approved tool arguments, returned
evidence, and the final report. It deliberately excludes private reasoning tokens.

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

Coverage includes successful structured decisions, dynamic tool selection, malformed output,
invalid tools and arguments, fabricated evidence, connection failures, timeouts, missing models,
retry exhaustion, duplicate tool prevention, budget termination, trace persistence, the original
mock investigations, and benchmark reporting.

## Current scope and limitations

- The generator and contracts model one retail domain and three suppliers.
- SQLite is suitable for this local MVP, not high-concurrency production workloads.
- Problematic records are intentionally sampled, so dimension impact is sample-scoped.
- Confidence labels express qualitative uncertainty, not calibrated probabilities.
- Small local models can remain inconsistent despite structured schemas and retries; inconclusive or
  failed reports are valid outcomes.
- RX 6600 XT acceleration is environment-dependent and must be verified locally.
