# BankGuard AI — Production-Grade Agentic Banking Operations Platform

> **LLM proposes → Policy determines → Human approves where required → Tool executes → Everything is audited.**

A complete, end-to-end agentic AI platform for banking payment investigation. Built as a monorepo with 10 independently demonstrable engineering modules — designed for Staff/Principal/Architect-level portfolio and interview use.

---

## Architecture

```
                         Users
                           │
                         HTTPS (Caddy)
                           │
                    EC2 t4g.xlarge (~$65/mo)
                           │
       ┌───────────────────┼─────────────────────┐
       │                   │                     │
   Streamlit UI        FastAPI                 React
                           │
                    Agent Harness (Runtime)
                           │
                       LangGraph
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
                     (validation, RBAC,
                    idempotency, audit)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
      PostgreSQL        Valkey           pgvector
     (cases, audit)   (working mem)   (semantic mem)
                           │
                   Amazon Bedrock (Nova Micro)
```

---

## 10 Engineering Modules

| Module | Location | Concept |
|---|---|---|
| **Agent Harness** | `harness/` | Budget, retries, circuit breaker, sandbox, duplicate detection |
| **Loop Engineering** | `agents/state.py`, `agents/supervisor/` | PLAN→ACT→OBSERVE→EVALUATE state machine, stopping conditions |
| **Context Engineering** | `context/` | Token-budgeted context contract, PII masking, relevance ranking |
| **Tool Gateway** | `tools/` | Pydantic contracts, structured errors, idempotency, RBAC |
| **Memory Architecture** | `memory/` | Working (Valkey), Episodic (PostgreSQL), Semantic (pgvector) |
| **Multi-Agent Orchestration** | `agents/` | LangGraph supervisor + 4 specialist agents |
| **Guardrails & HITL** | `guardrails/`, `approvals/` | OPA Rego policies, risk matrix, approval lifecycle |
| **Evaluations** | `evals/` | 15 test cases, 7-metric grader, unsafe action hard-override |
| **Observability** | `observability/` | OTel traces, Prometheus metrics, Langfuse, Loki |
| **FastAPI + Streamlit** | `apps/` | REST + WebSocket API, full operations portal UI |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- AWS credentials with Bedrock access (ap-south-1)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials and passwords
```

### 2. Start infrastructure

```bash
docker-compose up -d postgres valkey langfuse otel-collector prometheus grafana loki opa
```

### 3. Seed banking data

```bash
pip install -e ".[eval]"
python synthetic_data/seed_data.py
```

### 4. Ingest policies into vector store

```bash
python -c "
import asyncio, asyncpg
from memory.semantic.vector_store import SemanticMemory
from config import settings

async def main():
    db = await asyncpg.create_pool(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
    sm = SemanticMemory(db)
    n = await sm.ingest_all_policies()
    print(f'Ingested {n} policy chunks')
    await db.close()

asyncio.run(main())
"
```

### 5. Start the API

```bash
uvicorn apps.api.main:app --reload --port 8000
```

### 6. Start the UI

```bash
streamlit run apps/operations-ui/app.py --server.port 5173
```

### 7. Run evaluations

```bash
python evals/regression/run_evals.py
```

---

## Demo Scenario

Navigate to `http://localhost:5173` → **🔍 Investigate** tab.

Create a case:
- **Title:** IMPS payment debited, beneficiary not credited
- **Description:** Customer reports ₹25,000 debited but beneficiary did not receive payment via IMPS
- **Transaction Ref:** TXN-30001

Expected investigation output:
```
✓ Customer identity verified
✓ Transaction TXN-30001 retrieved — status: PENDING
✓ Payment rail events: last event BENEFICIARY_BANK_TIMEOUT
✓ Policy PAY-REC-101 retrieved
✓ Recent transactions checked

Root Cause: BENEFICIARY_BANK_TIMEOUT
Recommended Action: Wait for reconciliation window (24–48h). Do NOT retry.
Confidence: 85%
Requires human approval: No
```

Then try a reversal case — the platform will block it:
```
ACTION BLOCKED
Tool 'reverse_transaction' requires human approval.
Risk: CRITICAL
Submit an approval request.
```

---

## Cost Estimate (AWS Mumbai / ap-south-1)

| Component | $/month (24×7) |
|---|---:|
| EC2 t4g.xlarge | $65.41 |
| 60 GB gp3 EBS | $5.47 |
| Public IPv4 | $3.65 |
| S3 10 GB | $0.25 |
| Bedrock Nova Micro (1K cases) | ~$1.75 |
| Cognito (free tier) | $0 |
| All OSS services | $0 |
| **Total** | **~$80–90/mo** |

Development hours only (~176h/mo): **~$27–32/mo**

---

## Tech Stack

```
Agent:         LangGraph + LiteLLM → Amazon Bedrock Nova Micro/Lite
Validation:    Pydantic v2
API:           FastAPI + WebSockets
UI:            Streamlit
DB:            PostgreSQL 16 + pgvector (HNSW)
Cache:         Valkey (Redis-compatible OSS)
Embeddings:    BAAI/bge-small-en-v1.5 (sentence-transformers, CPU)
Workflows:     Temporal OSS
Auth:          Cognito (user pool) + OPA (tool authorization)
PII:           Microsoft Presidio
Tracing:       OpenTelemetry → Langfuse + Prometheus
Logs:          structlog (JSON) → Loki → Grafana
Containers:    Docker Compose
IaC:           Terraform (infrastructure/)
Retry:         Tenacity + pybreaker
```

---

## Repository Structure

```
bankguard-ai/
├── agents/          # LangGraph multi-agent graph + specialist agents
├── apps/
│   ├── api/         # FastAPI backend
│   └── operations-ui/ # Streamlit portal
├── approvals/       # HITL approval service
├── context/         # Context Builder + token budget + ranking
├── evals/           # Evaluation framework (datasets, graders, runner)
├── guardrails/      # RBAC, OPA Rego policies, PII filter
├── harness/         # Agent runtime: budget, retries, circuit breaker, sandbox
├── infrastructure/  # Docker, OTel, Prometheus, Loki, Caddy configs + Terraform
├── memory/          # Working (Valkey), Episodic (PG), Semantic (pgvector)
├── observability/   # Telemetry setup, Prometheus metrics, Langfuse
├── synthetic_data/  # Banking data generator + seeder
├── tools/           # Tool Gateway + 8 read + 5 write tools
├── config.py        # Centralised settings (pydantic-settings)
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```
