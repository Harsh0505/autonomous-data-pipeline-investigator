Retail Pipeline Guardian

An agentic AI system for investigating retail data-pipeline failures and safely testing repairs. It detects data-quality issues, gathers operational evidence, evaluates possible root causes, proposes constrained fixes, and verifies them against an isolated dataset.

Built with Python, LangGraph, Pydantic, SQLite, pandas, and Ollama. Runs locally without a paid API key.

How it works

Retail data → Quality checks → Incident
                                  ↓
                      Investigation agent
                    (tools + evidence + hypotheses)
                                  ↓
                        Root-cause report
                                  ↓
                        Repair-planning agent
                                  ↓
                         Human approval
                                  ↓
                      Isolated repair simulation
                                  ↓
                     Deterministic verification
                                  ↓
                       Recommend or reject

The investigation agent chooses read-only tools to inspect supplier submissions, pipeline logs, orchestration events, transformations, and configuration changes. A separate factual verifier prevents unsupported causal claims. The repair agent proposes only approved, parameterised operations. Original data is never modified, and the LLM cannot declare its own repair successful.

Example

In a reproducible duplicate-record scenario, the investigation identified two accepted submissions with the same supplier batch ID and checksum. The repair workflow proposed deduplication, removed five duplicated records in an isolated simulation (205 → 200), and verified that the uniqueness check passed without introducing new failures. This is a synthetic demonstration, not a production deployment.

Supported simulated repairs:

Deduplicate specified transaction IDs.

Correct supplier currency using its configured contract.

Activate a category backed by an approved configuration change.

Quick start (Windows PowerShell)

Requires Python 3.12+.

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

Run a reproducible incident and investigation:

$db = "artifacts\demo.db"
retail-guardian run --scenario duplicate_records --rows 500 --seed 42 --database $db --output-dir artifacts\demo
$incident = Get-Content artifacts\demo\duplicate_records-incident.json | ConvertFrom-Json
$investigation = retail-guardian investigate --provider mock --database $db --incident-id $incident.incident_id | ConvertFrom-Json

Propose, approve, and simulate a repair:

$repair = retail-guardian propose-repair --provider mock --database $db --investigation-id $investigation.investigation_id | ConvertFrom-Json
retail-guardian show-repair --database $db --repair-id $repair.repair_id
retail-guardian approve-repair --database $db --repair-id $repair.repair_id
retail-guardian simulate-repair --database $db --repair-id $repair.repair_id
retail-guardian repair-result --database $db --repair-id $repair.repair_id

The mock provider is a deterministic demonstration/testing provider, not an LLM. For live local reasoning, install Ollama, run ollama pull qwen3:4b, and use --provider ollama --model qwen3:4b for investigation or repair proposals.

Evaluation and tests

6/6 deterministic incident-regression cases passed with the real local qwen3:4b model in a recorded run. This measures workflow reliability on known incident types, not complex upstream causal reasoning.

A separate ten-case benchmark tests competing upstream causes and insufficient evidence. Local-model results vary; the initial recorded run achieved 2 correct diagnoses among 9 resolvable attempts, with no evidence-supported conclusions. Subsequent architectural fixes were verified through targeted tests, not a completed repeat of that benchmark.

145 automated tests passed at the Day 4 milestone; subsequent changes should be retested before updating this number.

python -m pytest -W error
python -m ruff check .
python -m ruff format --check .

Safety and limitations

Synthetic data and operational logs; SQLite storage is intended for local demonstration.

Repairs run only against isolated snapshots after explicit approval for simulation. No production-apply command exists.

The causal verifier supports nine defined mechanisms, and the repair library supports three operations.

Local LLMs may choose poor tools or fail to reach a supported diagnosis; the system records failures and inconclusive outcomes.

A successful data repair does not necessarily fix the upstream process that introduced the defect.

See the CLI --help commands and source modules under src/retail_guardian/ for the full investigation, benchmark, and repair interfaces.