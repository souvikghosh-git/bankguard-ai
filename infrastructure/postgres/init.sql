-- ============================================================
-- BankGuard AI — PostgreSQL initialization
-- Creates all schemas, tables, and extensions
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";         -- pgvector for semantic memory

-- ============================================================
-- SCHEMA: banking  (synthetic banking domain data)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS banking;

CREATE TABLE IF NOT EXISTS banking.customers (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_ref    VARCHAR(20) UNIQUE NOT NULL,   -- e.g. CUST-10001
    full_name       VARCHAR(200) NOT NULL,
    email           VARCHAR(200),
    phone           VARCHAR(20),
    kyc_status      VARCHAR(20) DEFAULT 'VERIFIED',
    risk_category   VARCHAR(20) DEFAULT 'LOW',     -- LOW | MEDIUM | HIGH
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS banking.accounts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_ref     VARCHAR(20) UNIQUE NOT NULL,   -- e.g. ACC-20001
    customer_id     UUID REFERENCES banking.customers(id),
    account_type    VARCHAR(30) NOT NULL,           -- SAVINGS | CURRENT | FD
    balance         NUMERIC(18,2) NOT NULL DEFAULT 0,
    currency        CHAR(3) DEFAULT 'INR',
    status          VARCHAR(20) DEFAULT 'ACTIVE',  -- ACTIVE | FROZEN | CLOSED
    ifsc_code       VARCHAR(15),
    bank_name       VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS banking.transactions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_ref     VARCHAR(30) UNIQUE NOT NULL,  -- e.g. TXN-30001
    debit_account_id    UUID REFERENCES banking.accounts(id),
    credit_account_id   UUID,                          -- may be external
    amount              NUMERIC(18,2) NOT NULL,
    currency            CHAR(3) DEFAULT 'INR',
    transaction_type    VARCHAR(30) NOT NULL,          -- NEFT | IMPS | UPI | RTGS
    status              VARCHAR(30) NOT NULL,          -- PENDING | COMPLETED | FAILED | REVERSED
    reference_number    VARCHAR(50),                   -- bank/rail reference
    description         TEXT,
    initiated_at        TIMESTAMPTZ DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS banking.payment_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id  UUID REFERENCES banking.transactions(id),
    event_type      VARCHAR(50) NOT NULL,
    event_code      VARCHAR(50),
    event_message   TEXT,
    payment_rail    VARCHAR(20),                  -- NEFT | IMPS | UPI | RTGS
    occurred_at     TIMESTAMPTZ DEFAULT NOW(),
    source_system   VARCHAR(50),
    raw_payload     JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS banking.beneficiaries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id     UUID REFERENCES banking.customers(id),
    name            VARCHAR(200) NOT NULL,
    account_number  VARCHAR(30),
    ifsc_code       VARCHAR(15),
    bank_name       VARCHAR(100),
    verified        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS banking.policies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_ref      VARCHAR(30) UNIQUE NOT NULL,   -- e.g. PAY-REC-102
    title           VARCHAR(300) NOT NULL,
    category        VARCHAR(50),                    -- PAYMENT | REFUND | LIMIT | COMPLIANCE
    content         TEXT NOT NULL,
    effective_date  DATE,
    version         VARCHAR(10) DEFAULT '1.0',
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- SCHEMA: agent  (agent runtime state)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS agent;

CREATE TABLE IF NOT EXISTS agent.cases (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_ref        VARCHAR(30) UNIQUE NOT NULL,   -- e.g. CASE-9845
    title           VARCHAR(500) NOT NULL,
    description     TEXT NOT NULL,
    status          VARCHAR(30) DEFAULT 'OPEN',    -- OPEN | INVESTIGATING | PENDING_APPROVAL | RESOLVED | CLOSED | ESCALATED
    priority        VARCHAR(20) DEFAULT 'MEDIUM',  -- LOW | MEDIUM | HIGH | CRITICAL
    customer_id     UUID REFERENCES banking.customers(id),
    transaction_id  UUID REFERENCES banking.transactions(id),
    assigned_to     VARCHAR(100),
    root_cause      TEXT,
    resolution      TEXT,
    confidence      FLOAT,
    requires_human_approval BOOLEAN DEFAULT FALSE,
    action_risk_level       VARCHAR(20),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agent.agent_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_ref         VARCHAR(50) UNIQUE NOT NULL,
    case_id         UUID REFERENCES agent.cases(id),
    agent_id        VARCHAR(100),
    model_id        VARCHAR(100),
    status          VARCHAR(30) DEFAULT 'RUNNING', -- RUNNING | COMPLETED | FAILED | CANCELLED
    iteration_count INT DEFAULT 0,
    tool_call_count INT DEFAULT 0,
    input_tokens    INT DEFAULT 0,
    output_tokens   INT DEFAULT 0,
    cost_usd        NUMERIC(10,6) DEFAULT 0,
    duration_ms     INT,
    stopping_reason VARCHAR(50),
    final_output    JSONB,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agent.tool_calls (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          UUID REFERENCES agent.agent_runs(id),
    tool_name       VARCHAR(100) NOT NULL,
    tool_input      JSONB NOT NULL,
    tool_output     JSONB,
    status          VARCHAR(20) DEFAULT 'SUCCESS', -- SUCCESS | ERROR | TIMEOUT | BLOCKED
    error_code      VARCHAR(50),
    duration_ms     INT,
    attempt_number  INT DEFAULT 1,
    called_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent.case_notes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID REFERENCES agent.cases(id),
    note_type       VARCHAR(30) DEFAULT 'INVESTIGATION',  -- INVESTIGATION | ACTION | RESOLUTION | ESCALATION
    content         TEXT NOT NULL,
    created_by      VARCHAR(100) DEFAULT 'agent',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent.approvals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    approval_ref    VARCHAR(50) UNIQUE NOT NULL,
    case_id         UUID REFERENCES agent.cases(id),
    run_id          UUID REFERENCES agent.agent_runs(id),
    action_type     VARCHAR(100) NOT NULL,
    action_payload  JSONB NOT NULL,
    risk_level      VARCHAR(20) NOT NULL,           -- LOW | MEDIUM | HIGH | CRITICAL
    status          VARCHAR(20) DEFAULT 'PENDING',  -- PENDING | APPROVED | REJECTED | EXPIRED
    requested_by    VARCHAR(100),
    reviewed_by     VARCHAR(100),
    review_notes    TEXT,
    requested_at    TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ
);

-- ============================================================
-- SCHEMA: memory  (agent memory layers)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS memory;

CREATE TABLE IF NOT EXISTS memory.episodic (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         UUID REFERENCES agent.cases(id),
    run_id          UUID REFERENCES agent.agent_runs(id),
    event_type      VARCHAR(50),
    summary         TEXT NOT NULL,
    evidence        JSONB DEFAULT '[]',
    outcome         VARCHAR(50),
    occurred_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Semantic memory: policies and operational knowledge as embeddings
CREATE TABLE IF NOT EXISTS memory.semantic_store (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type     VARCHAR(50) NOT NULL,          -- POLICY | SOP | CASE_PATTERN
    source_ref      VARCHAR(50),
    content         TEXT NOT NULL,
    embedding       vector(384),                   -- BGE-small produces 384-dim
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast ANN search
CREATE INDEX IF NOT EXISTS idx_semantic_embedding
    ON memory.semantic_store USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- SCHEMA: eval  (evaluation framework)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS eval;

CREATE TABLE IF NOT EXISTS eval.test_cases (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_ref        VARCHAR(50) UNIQUE NOT NULL,
    category        VARCHAR(50),
    scenario        TEXT NOT NULL,
    input_case      JSONB NOT NULL,
    expected_root_cause     VARCHAR(200),
    expected_action         VARCHAR(200),
    expected_tools          JSONB DEFAULT '[]',
    is_unsafe_action        BOOLEAN DEFAULT FALSE,
    requires_human_approval BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval.eval_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_ref         VARCHAR(50) UNIQUE NOT NULL,
    model_id        VARCHAR(100),
    git_commit      VARCHAR(50),
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    summary         JSONB
);

CREATE TABLE IF NOT EXISTS eval.eval_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    eval_run_id     UUID REFERENCES eval.eval_runs(id),
    test_case_id    UUID REFERENCES eval.test_cases(id),
    agent_run_id    UUID REFERENCES agent.agent_runs(id),
    root_cause_correct      BOOLEAN,
    action_correct          BOOLEAN,
    unsafe_action_taken     BOOLEAN DEFAULT FALSE,
    human_escalation_correct BOOLEAN,
    policy_applied_correctly BOOLEAN,
    evidence_grounded       BOOLEAN,
    tool_calls_count        INT,
    unnecessary_tool_calls  INT DEFAULT 0,
    total_cost_usd          NUMERIC(10,6),
    duration_ms             INT,
    outcome_score           FLOAT,
    grader_notes            TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_transactions_ref ON banking.transactions(transaction_ref);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON banking.transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_debit_account ON banking.transactions(debit_account_id);
CREATE INDEX IF NOT EXISTS idx_payment_events_txn ON banking.payment_events(transaction_id);
CREATE INDEX IF NOT EXISTS idx_cases_ref ON agent.cases(case_ref);
CREATE INDEX IF NOT EXISTS idx_cases_customer ON agent.cases(customer_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_case ON agent.agent_runs(case_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON agent.tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON agent.approvals(status);
