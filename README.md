# BankGuard AI — Production-Grade Agentic Banking Operations Platform

> **LLM proposes → Policy determines → Human approves where required → Tool executes → Everything is audited.**

A complete, end-to-end agentic AI platform for banking payment investigation.
Built as a monorepo with **10 independently demonstrable engineering modules** — designed for Staff/Principal/Architect-level portfolio and interview discussion.

---

## Architecture

```
                         Users
                           │
                     HTTPS (Caddy)
                           │
                    EC2 t4g.xlarge (~$80/mo)
                      Docker Compose
                           │
       ┌───────────────────┼─────────────────────┐
       │                   │                     │
  Streamlit UI         FastAPI               (React —
  (operations-ui)    (REST + WS)           future scope)
                           │
                    Agent Harness
                    (budget, retry,
                  circuit-breaker, sandbox)
                           │
                       LangGraph
                    Supervisor Graph
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    Transaction         Policy         Resolution
       Agent             Agent            Agent
          │                │                │
          └────────────────┼────────────────┘
                           │
                    Reviewer Agent
                           │
                      Tool Gateway
              (Pydantic validation · RBAC · OPA
               idempotency · audit · timeout)
                           │
          ┌────────────────┼──────────────────┐
          │                │                  │
      PostgreSQL        Valkey            pgvector
   (cases · audit)  (working mem)     (semantic mem)
                           │
                  ┌────────┴────────┐
                  │                 │
           Temporal OSS        Bedrock
           (HITL durable     (Nova Micro /
            workflows)         Nova Lite)
                           │
                  Observability stack
           (OTel → Langfuse · Prometheus · Loki → Grafana)
```

---

## 10 Engineering Modules

| # | Module | Location | What it demonstrates |
|---|---|---|---|
| 1 | **Agent Harness** | `harness/` | Budget (tokens/cost/iterations/parallel tools), exponential-backoff retry, pybreaker circuit breaker, sandbox modes, duplicate-call detection |
| 2 | **Loop Engineering** | `agents/state.py`, `agents/supervisor/` | Explicit PLAN→ACT→OBSERVE→EVALUATE state machine, 8 named stopping conditions, confidence floor, evidence delta tracking |
| 3 | **Context Engineering** | `context/` | Token-budgeted context contract, PII masking via Presidio, policy + transaction relevance ranking |
| 4 | **Tool Gateway** | `tools/` | Pydantic input/output contracts, structured errors, RBAC permission check (local + OPA), idempotency via Valkey, per-call audit persistence |
| 5 | **Memory Architecture** | `memory/` | Working memory (Valkey, TTL 4 h), Episodic memory (PostgreSQL), Semantic memory (pgvector HNSW + BGE-small embeddings) |
| 6 | **Multi-Agent Orchestration** | `agents/` | LangGraph supervisor + Transaction Investigator, Policy, Resolution, Reviewer specialist agents |
| 7 | **Guardrails & HITL** | `guardrails/`, `approvals/`, `workflows/temporal/` | OPA Rego policy-as-code, risk matrix, Temporal durable approval workflow with process-restart safety |
| 8 | **Observability** | `observability/` | OTel trace propagation (run_id linked to trace context), Prometheus counters/histograms wired to every agent event, Langfuse LLM tracing, structlog→Loki |
| 9 | **Evaluation Framework** | `evals/` | 15 standard + 10 adversarial test cases, 7-metric outcome grader (unsafe action = score 0), live agent eval runner |
| 10 | **API & Operations Portal** | `apps/` | FastAPI REST + WebSocket investigation streaming, Streamlit 6-page portal (cases, HITL approvals, eval runner, metrics) |

---

## Technology Stack

```
Agent framework:   LangGraph 0.2 + LiteLLM (provider-agnostic)
LLM:               Amazon Bedrock Nova Micro (default) / Nova Lite (fallback)
                   OpenAI gpt-4o-mini supported for local dev via LiteLLM
API:               FastAPI + Uvicorn (async)
UI:                Streamlit (operations portal)
Validation:        Pydantic v2
Database:          PostgreSQL 16 + pgvector (HNSW index, 384-dim BGE embeddings)
Cache:             Valkey 7 (Redis-compatible OSS)
Embeddings:        BAAI/bge-small-en-v1.5 (sentence-transformers, CPU)
Durable workflow:  Temporal OSS 1.25 (HITL approval, process-restart safety)
Authorization:     OPA 0.68 (Rego policy-as-code)
PII filtering:     Microsoft Presidio
Observability:     OpenTelemetry SDK → OTel Collector → Langfuse + Prometheus
Logging:           structlog (JSON) → Loki → Grafana
Retry:             Tenacity + pybreaker
Containers:        Docker Compose (single EC2 node)
CI/CD:             GitHub Actions (lint, unit tests, dry-run eval gate)
```

---

## Quick Start

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- Python 3.11+
- One of: OpenAI API key **or** AWS account with Bedrock Nova Micro enabled in `ap-south-1`

### 1. Configure environment

```bash
# For local dev with OpenAI (no AWS required):
cp .env.local.example .env
# Edit .env → set OPENAI_API_KEY=sk-...

# For Bedrock on EC2 (use IAM Instance Role — no keys in .env):
cp .env.example .env
# Leave AWS_ACCESS_KEY_ID blank; the SDK credential chain uses the Instance Role
```

### 2. Start infrastructure

```bash
# Local dev (PostgreSQL, Valkey, Langfuse, Prometheus, Grafana, OPA)
docker-compose -f docker-compose.local.yml up -d

# Wait for postgres to be ready, then seed data
python synthetic_data/seed_data.py
```

### 3. Ingest policies into pgvector

```bash
python -c "
import asyncio, asyncpg
from memory.semantic.vector_store import SemanticMemory
from config import settings

async def main():
    url = settings.database_url.replace('postgresql+asyncpg://', 'postgresql://')
    db = await asyncpg.create_pool(url)
    n = await SemanticMemory(db).ingest_all_policies()
    print(f'Ingested {n} policy chunks')
    await db.close()

asyncio.run(main())
"
```

### 4. Run API + UI

```bash
# Terminal 1
uvicorn apps.api.main:app --reload --port 8000

# Terminal 2
streamlit run apps/operations-ui/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 5. Run evaluations

```bash
# Dry-run (mock agent, tests grader logic)
python evals/regression/run_evals.py

# Adversarial suite
python evals/adversarial/run_adversarial.py

# Live (requires running DB + LLM)
python evals/regression/run_evals.py --live
```

---

## Demo Scenario

Go to **🔍 Investigate** → create a case:

- **Title:** IMPS payment debited, beneficiary not credited
- **Transaction Ref:** TXN-30001

Expected agent output:
```
✓ Transaction TXN-30001 retrieved — status: PENDING
✓ Last event: BENEFICIARY_BANK_TIMEOUT
✓ Policy PAY-REC-101 retrieved: do not retry during reconciliation window
✓ Case note created

Root Cause:   BENEFICIARY_BANK_TIMEOUT
Action:       Wait 24–48h reconciliation window. Do NOT retry.
Confidence:   85%
Risk Level:   LOW
Approval:     Not required
```

Then investigate a reversal case — the gateway blocks it:
```
ACTION BLOCKED
Tool 'reverse_transaction' requires human approval.
Risk: CRITICAL
An approval request has been submitted: APR-XXXXXXXX
Waiting for Risk Officer decision via Temporal workflow…
```

---

## AWS Credentials — Important

**Never put static AWS credentials in `.env` on EC2.**

Use the AWS credential chain in this order:

1. **EC2 Instance Role** (production) — attach an IAM role to your EC2 instance with `bedrock:InvokeModel` permission. The SDK finds it automatically via IMDS.
2. **`~/.aws/credentials` profile** (local dev with Bedrock) — `aws configure --profile bankguard`
3. **Environment variables** (CI/CD only) — `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` as GitHub Secrets, never in files.
4. **OpenAI via LiteLLM** (local dev without AWS) — set `OPENAI_API_KEY` in `.env`.

The `.env.example` in this repo has `AWS_ACCESS_KEY_ID` commented out deliberately.

---

## Cost Estimate (AWS Mumbai / ap-south-1, 24×7)

| Component | $/month |
|---|---:|
| EC2 t4g.xlarge (4 vCPU / 16 GB) | $65.41 |
| 60 GB gp3 EBS | $5.47 |
| Public IPv4 | $3.65 |
| S3 10 GB | $0.25 |
| Bedrock Nova Micro (1 K investigations) | ~$1.75 |
| Cognito (≤10 K MAU free tier) | $0 |
| All OSS containers | $0 |
| **Total** | **~$78–90/mo** |

Development hours only (~176 h/mo): **~$27–32/mo**

---

## Repository Structure

```
bankguard-ai/
├── agents/                # LangGraph graph + 4 specialist agents + runner
│   ├── supervisor/        # StateGraph topology, stopping conditions
│   ├── transaction_agent/ # Evidence gathering, payment event heuristics
│   ├── policy_agent/      # Policy retrieval and case history
│   ├── resolution_agent/  # LLM + rule-based root-cause + action proposal
│   ├── reviewer_agent/    # Safety/evidence/policy compliance review
│   ├── state.py           # AgentState TypedDict, LoopState, StopReason
│   └── runner.py          # AgentRunner — orchestrates harness + graph + memory
├── apps/
│   ├── api/               # FastAPI: cases, investigations (WS), approvals
│   └── operations-ui/     # Streamlit portal (6 pages)
├── approvals/             # HITL approval DB lifecycle (create/approve/reject)
├── context/               # Context Builder, token budget, PII filter, ranking
├── evals/
│   ├── datasets/          # 15 standard test cases (JSON)
│   ├── adversarial/       # 10 adversarial + failure scenarios
│   ├── graders/           # OutcomeGrader (7 metrics, unsafe=0 override)
│   └── regression/        # Eval runner (dry-run + live)
├── guardrails/
│   ├── permissions/       # RBAC engine (local + OPA delegation)
│   ├── pii/               # Presidio PII masking
│   └── policy_engine/rego # OPA Rego: bankguard_tool_access.rego
├── harness/               # Runtime, budget, retries, circuit breaker, sandbox
├── infrastructure/
│   ├── caddy/             # Caddyfile (TLS reverse proxy)
│   ├── loki/              # Loki config
│   ├── otel/              # OTel Collector config
│   ├── postgres/          # init.sql (4 schemas, 12 tables, pgvector)
│   └── prometheus/        # Prometheus scrape config
├── memory/
│   ├── working/           # Valkey-backed per-run state (TTL 4 h)
│   ├── episodic/          # PostgreSQL case episode store
│   └── semantic/          # pgvector HNSW + BGE-small embeddings
├── observability/         # OTel setup, Prometheus metrics, Langfuse, structlog
├── synthetic_data/        # Banking data generator + DB seeder
├── tools/                 # Tool Gateway + 8 read + 3 write tools
├── workflows/
│   └── temporal/          # Temporal worker, HITL workflow, approval activities
├── scripts/               # start_local.sh
├── config.py              # Centralised pydantic-settings
├── docker-compose.yml     # Full EC2 stack (hardened — internal services on 127.0.0.1)
├── docker-compose.local.yml # Local dev (infra only, no Caddy)
├── pyproject.toml
└── .env.example           # No static credentials; see AWS section above
```

---

## Security Hardening Notes

- All internal services (Postgres, Valkey, OPA, Temporal, Prometheus, Loki, OTLP) bind to `127.0.0.1` on the host — not reachable from outside the EC2 instance.
- Only Caddy exposes ports 80/443 to the internet; traffic reaches services via Docker internal DNS.
- No static AWS credentials in any committed file.
- PII is masked via Presidio before any text reaches the LLM.
- Tool Gateway blocks CRITICAL/HIGH-risk actions unconditionally; they only execute after a Temporal-durable human approval workflow completes.
- OPA enforces role-based tool access as a second layer on top of local RBAC.
