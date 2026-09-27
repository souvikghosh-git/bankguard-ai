"""
BankGuard AI — Synthetic data generator.
Creates realistic banking data: customers, accounts, transactions,
payment events, beneficiaries and policies for development/eval.

Run:  python synthetic_data/seed_data.py
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import structlog

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Domain data
# ──────────────────────────────────────────────────────────────────────────────

BANKS = [
    ("HDFC Bank", "HDFC0001234"),
    ("ICICI Bank", "ICIC0002345"),
    ("SBI", "SBIN0003456"),
    ("Axis Bank", "UTIB0004567"),
    ("Kotak Mahindra", "KKBK0005678"),
    ("Yes Bank", "YESB0006789"),
    ("Punjab National Bank", "PUNB0007890"),
    ("Bank of Baroda", "BARB0008901"),
]

PAYMENT_RAILS = ["IMPS", "NEFT", "UPI", "RTGS"]

TRANSACTION_TYPES = ["TRANSFER", "PAYMENT", "REFUND", "REVERSAL"]

FIRST_NAMES = [
    "Arjun",
    "Priya",
    "Rahul",
    "Anita",
    "Vikram",
    "Sunita",
    "Deepak",
    "Kavitha",
    "Rajesh",
    "Meena",
    "Suresh",
    "Pooja",
    "Arun",
    "Rekha",
    "Ganesh",
]
LAST_NAMES = [
    "Sharma",
    "Patel",
    "Kumar",
    "Gupta",
    "Singh",
    "Nair",
    "Reddy",
    "Iyer",
    "Pillai",
    "Verma",
]

# Payment event sequences per scenario
SCENARIO_EVENTS = {
    "BENEFICIARY_BANK_TIMEOUT": [
        ("PAYMENT_INITIATED", "Payment initiated on {rail}"),
        ("DEBIT_SUCCESSFUL", "Debit of ₹{amount} successful from account"),
        ("ROUTING_STARTED", "Payment routed to beneficiary bank {bank}"),
        ("BENEFICIARY_BANK_TIMEOUT", "Timeout waiting for acknowledgement from {bank}"),
        ("PENDING_RECONCILIATION", "Payment held pending reconciliation"),
    ],
    "COMPLETED": [
        ("PAYMENT_INITIATED", "Payment initiated on {rail}"),
        ("DEBIT_SUCCESSFUL", "Debit of ₹{amount} successful from account"),
        ("ROUTING_STARTED", "Payment routed to beneficiary bank {bank}"),
        ("CREDIT_CONFIRMED", "Credit confirmed by {bank}"),
        ("PAYMENT_COMPLETED", "Payment completed successfully"),
    ],
    "INSUFFICIENT_FUNDS": [
        ("PAYMENT_INITIATED", "Payment initiated on {rail}"),
        ("DEBIT_FAILED", "Insufficient funds in account"),
        ("PAYMENT_REJECTED", "Payment rejected: INSUFFICIENT_FUNDS"),
    ],
    "DUPLICATE_DETECTED": [
        ("PAYMENT_INITIATED", "Payment initiated on {rail}"),
        ("DUPLICATE_CHECK_FAILED", "Duplicate transaction detected within 60 seconds"),
        ("PAYMENT_REJECTED", "Payment rejected: DUPLICATE_TRANSACTION"),
    ],
    "RAIL_UNAVAILABLE": [
        ("PAYMENT_INITIATED", "Payment initiated on {rail}"),
        ("RAIL_UNAVAILABLE", "{rail} currently unavailable for processing"),
        ("FALLBACK_ATTEMPTED", "Attempting fallback rail"),
        ("FALLBACK_FAILED", "All payment rails unavailable"),
        ("PAYMENT_FAILED", "Payment failed: RAIL_UNAVAILABLE"),
    ],
}

POLICIES = [
    {
        "policy_ref": "PAY-REC-101",
        "title": "Payment Reconciliation Policy — IMPS/NEFT Timeout",
        "category": "PAYMENT",
        "content": """
## Payment Reconciliation Policy — IMPS/NEFT Timeout (PAY-REC-101)

### Scope
Applies to all IMPS and NEFT transactions where the beneficiary bank has not confirmed
credit within the standard settlement window.

### Standard Settlement Windows
- IMPS: 30 minutes from initiation
- NEFT: Settled in hourly batches (next batch + 2 hours buffer)
- RTGS: 30 minutes during operating hours

### Procedure
1. If debit is confirmed but beneficiary credit is unconfirmed within the settlement window:
   - DO NOT initiate a second transfer
   - Wait for the full reconciliation window (24 hours for IMPS, 48 hours for NEFT)
   - Check NPCI/payment rail status before any action

2. If reconciliation window expires without confirmation:
   - Initiate refund to originating account
   - Log incident in case management system
   - Notify customer within 1 business day

### Prohibited Actions
- Initiating duplicate payment while original is in PENDING_RECONCILIATION state
- Reversing transaction before reconciliation window expires
- Crediting customer account before confirming debit reversal from payment rail

### Authorization
Fee waivers up to ₹500 can be approved by Operations Analyst.
Fee waivers above ₹500 require Risk Officer approval.
""",
        "effective_date": "2024-01-01",
    },
    {
        "policy_ref": "PAY-REF-201",
        "title": "Fee Refund Policy — Transaction Errors",
        "category": "REFUND",
        "content": """
## Fee Refund Policy — Transaction Errors (PAY-REF-201)

### Scope
Governs refunds of transaction fees charged during failed or erroneous payments.

### Automatic Refund Eligibility
Fee refund is automatically eligible when:
- Transaction failed due to bank/system error (not customer error)
- Duplicate charge confirmed
- Payment rail outage caused failure

### Refund Limits by Role
| Role | Max Refund Without Approval |
|---|---|
| Operations Analyst | ₹500 |
| Senior Analyst | ₹2,000 |
| Risk Officer | ₹10,000 |
| Branch Manager | Unlimited with audit log |

### Process
1. Verify transaction failure reason
2. Confirm fee was actually charged
3. Obtain required approval level
4. Process refund within 2 business days
5. Notify customer via registered channel

### Prohibited
- Refunding fees for customer-initiated cancellations
- Processing refund without audit trail
""",
        "effective_date": "2024-01-01",
    },
    {
        "policy_ref": "PAY-LIM-301",
        "title": "Transaction Limit Policy — Retail Banking",
        "category": "LIMIT",
        "content": """
## Transaction Limit Policy — Retail Banking (PAY-LIM-301)

### Per-Transaction Limits
| Rail | Per Transaction | Per Day |
|---|---|---|
| IMPS | ₹5,00,000 | ₹10,00,000 |
| NEFT | ₹10,00,000 | ₹20,00,000 |
| UPI | ₹1,00,000 | ₹2,00,000 |
| RTGS | ₹2,00,000 (min) | ₹50,00,000 |

### Breach Protocol
1. Transaction exceeding limit must be rejected at initiation
2. Customer must be notified of limit breach
3. For legitimate high-value needs, customer can apply for temporary limit enhancement
4. Limit enhancement requires KYC verification and Risk Officer approval

### Fraud Alert Thresholds
Transactions above 10× the customer's 30-day average trigger fraud review.
""",
        "effective_date": "2024-01-01",
    },
    {
        "policy_ref": "COMP-AML-401",
        "title": "AML Transaction Monitoring Policy",
        "category": "COMPLIANCE",
        "content": """
## AML Transaction Monitoring Policy (COMP-AML-401)

### Scope
All transactions processed through the bank's payment systems.

### Monitoring Thresholds
- Transactions ≥ ₹10,00,000 trigger automatic AML review
- Multiple transactions totalling ≥ ₹10,00,000 in 24 hours trigger review
- Transactions to high-risk jurisdictions trigger enhanced review

### Agent Actions Under AML Review
- READ operations (transaction details, account info): ALLOWED
- Creating case notes: ALLOWED
- Initiating refunds during AML hold: PROHIBITED
- Releasing frozen accounts: REQUIRES compliance officer approval
- Any action on flagged accounts: REQUIRES compliance officer approval

### Escalation
AML-flagged cases must be escalated to Compliance Officer within 4 hours.
Do NOT notify customer of AML investigation.
""",
        "effective_date": "2024-01-01",
    },
    {
        "policy_ref": "OPS-FRZ-501",
        "title": "Account Freeze and Unfreeze Policy",
        "category": "OPERATIONS",
        "content": """
## Account Freeze and Unfreeze Policy (OPS-FRZ-501)

### Freeze Authorization Levels
| Reason | Authorization Required |
|---|---|
| Fraud suspected | Operations Analyst (temporary, max 24h) |
| AML triggered | Compliance Officer |
| Court order | Legal + Senior Management |
| Customer request | Operations Analyst |

### Unfreeze Authorization
Account unfreeze ALWAYS requires human approval.
No agent may unfreeze an account autonomously.

### Procedure
1. Document reason for freeze with evidence
2. Obtain required authorization
3. Record freeze in case management system
4. Notify customer (except AML cases)
5. Set review date (maximum 30 days)
""",
        "effective_date": "2024-01-01",
    },
    {
        "policy_ref": "PAY-REV-601",
        "title": "Payment Reversal Policy",
        "category": "PAYMENT",
        "content": """
## Payment Reversal Policy (PAY-REV-601)

### Eligibility for Reversal
A payment is eligible for reversal when:
- Wrong beneficiary credited (confirmed within 24 hours)
- Duplicate payment confirmed
- System error caused erroneous debit
- Fraudulent transaction confirmed

### Reversal Authorization
| Amount | Authorization |
|---|---|
| Up to ₹10,000 | Operations Analyst |
| ₹10,001 – ₹1,00,000 | Senior Analyst + Risk Officer |
| Above ₹1,00,000 | Risk Officer + Branch Manager |

### MANDATORY: No reversal without human approval.
Automated reversal is STRICTLY PROHIBITED regardless of amount.

### Process
1. Confirm reversal eligibility
2. Obtain appropriate authorization (documented)
3. Verify beneficiary bank can process reversal
4. Initiate reversal with unique idempotency key
5. Monitor until confirmed
6. Update case with outcome
""",
        "effective_date": "2024-01-01",
    },
    {
        "policy_ref": "OPS-NOTIF-701",
        "title": "Customer Notification Policy",
        "category": "OPERATIONS",
        "content": """
## Customer Notification Policy (OPS-NOTIF-701)

### Notification Requirements
| Event | Timeline | Channel |
|---|---|---|
| Transaction failure | Immediate | SMS + App |
| Pending resolution | 2 hours | SMS |
| Case created | 1 business day | Email |
| Case resolved | Immediate | SMS + Email |
| Account action | Before execution | SMS |

### Agent Notification Permissions
- Draft notifications: ALLOWED (requires human review before send)
- Send routine transaction status: ALLOWED (automated)
- Send sensitive account actions: REQUIRES human approval
- Send AML-related communications: PROHIBITED

### Content Rules
- Never disclose internal investigation details
- Never mention regulatory/compliance triggers
- Use approved notification templates only
""",
        "effective_date": "2024-01-01",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


def _past(days: int = 0, hours: int = 0, minutes: int = 0) -> datetime:
    return _now() - timedelta(days=days, hours=hours, minutes=minutes)


def gen_customers(n: int = 20) -> list[dict[str, Any]]:
    customers = []
    for i in range(n):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        customers.append(
            {
                "id": str(uuid.uuid4()),
                "customer_ref": f"CUST-{10001 + i}",
                "full_name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}@example.com",
                "phone": f"+91 9{random.randint(100000000, 999999999)}",
                "kyc_status": random.choices(["VERIFIED", "PENDING", "REJECTED"], weights=[90, 8, 2])[0],
                "risk_category": random.choices(["LOW", "MEDIUM", "HIGH"], weights=[75, 20, 5])[0],
            }
        )
    return customers


def gen_accounts(customers: list[dict]) -> list[dict[str, Any]]:
    accounts = []
    idx = 0
    for cust in customers:
        n_accounts = random.randint(1, 3)
        for _ in range(n_accounts):
            bank, ifsc = random.choice(BANKS)
            accounts.append(
                {
                    "id": str(uuid.uuid4()),
                    "account_ref": f"ACC-{20001 + idx}",
                    "customer_id": cust["id"],
                    "account_type": random.choice(["SAVINGS", "CURRENT", "SAVINGS"]),
                    "balance": float(round(random.uniform(1000, 500000), 2)),
                    "currency": "INR",
                    "status": random.choices(["ACTIVE", "FROZEN", "CLOSED"], weights=[92, 5, 3])[0],
                    "ifsc_code": ifsc,
                    "bank_name": bank,
                }
            )
            idx += 1
    return accounts


def gen_transactions(accounts: list[dict], n: int = 100) -> tuple[list[dict], list[dict]]:
    """Returns (transactions, payment_events)."""
    active_accounts = [a for a in accounts if a["status"] == "ACTIVE"]
    transactions = []
    events = []

    scenarios = list(SCENARIO_EVENTS.keys())
    scenario_weights = [30, 40, 10, 10, 10]  # % distribution

    for i in range(n):
        src = random.choice(active_accounts)
        dst = random.choice([a for a in active_accounts if a["id"] != src["id"]])
        rail = random.choice(PAYMENT_RAILS)
        scenario = random.choices(scenarios, weights=scenario_weights)[0]
        amount = round(random.uniform(500, 100000), 2)

        # Map scenario to transaction status
        status_map = {
            "BENEFICIARY_BANK_TIMEOUT": "PENDING",
            "COMPLETED": "COMPLETED",
            "INSUFFICIENT_FUNDS": "FAILED",
            "DUPLICATE_DETECTED": "FAILED",
            "RAIL_UNAVAILABLE": "FAILED",
        }
        txn_status = status_map[scenario]

        initiated = _past(days=random.randint(0, 30), hours=random.randint(0, 23))
        txn_id = str(uuid.uuid4())
        txn_ref = f"TXN-{30001 + i}"

        transactions.append(
            {
                "id": txn_id,
                "transaction_ref": txn_ref,
                "debit_account_id": src["id"],
                "credit_account_id": dst["id"],
                "amount": amount,
                "currency": "INR",
                "transaction_type": "TRANSFER",
                "status": txn_status,
                "reference_number": f"NPCI{random.randint(100000000, 999999999)}",
                "description": f"{rail} transfer to {dst['account_ref']}",
                "initiated_at": initiated,
                "completed_at": initiated + timedelta(minutes=random.randint(1, 30))
                if txn_status == "COMPLETED"
                else None,
                "metadata": {
                    "payment_rail": rail,
                    "scenario": scenario,
                    "beneficiary_bank": dst["bank_name"],
                    "beneficiary_ifsc": dst["ifsc_code"],
                },
            }
        )

        # Generate matching payment events
        template_events = SCENARIO_EVENTS[scenario]
        for j, (code, msg_tpl) in enumerate(template_events):
            msg = msg_tpl.format(rail=rail, amount=f"{amount:,.2f}", bank=dst["bank_name"])
            events.append(
                {
                    "id": str(uuid.uuid4()),
                    "transaction_id": txn_id,
                    "event_type": code,
                    "event_code": code,
                    "event_message": msg,
                    "payment_rail": rail,
                    "occurred_at": initiated + timedelta(seconds=j * 15),
                    "source_system": f"{rail}-GATEWAY",
                    "raw_payload": {"code": code, "rail": rail, "amount": amount},
                }
            )

    return transactions, events


# ──────────────────────────────────────────────────────────────────────────────
# Seeder
# ──────────────────────────────────────────────────────────────────────────────


async def seed(db_url: str) -> None:
    log.info("connecting_to_database", url=db_url.split("@")[-1])
    conn = await asyncpg.connect(db_url)

    try:
        log.info("generating_customers")
        customers = gen_customers(20)

        log.info("generating_accounts")
        accounts = gen_accounts(customers)

        log.info("generating_transactions")
        transactions, events = gen_transactions(accounts, n=100)

        # ── Insert customers ──────────────────────────────────────
        await conn.executemany(
            """
            INSERT INTO banking.customers
                (id, customer_ref, full_name, email, phone, kyc_status, risk_category)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (customer_ref) DO NOTHING
            """,
            [
                (
                    c["id"],
                    c["customer_ref"],
                    c["full_name"],
                    c["email"],
                    c["phone"],
                    c["kyc_status"],
                    c["risk_category"],
                )
                for c in customers
            ],
        )
        log.info("inserted_customers", count=len(customers))

        # ── Insert accounts ───────────────────────────────────────
        await conn.executemany(
            """
            INSERT INTO banking.accounts
                (id, account_ref, customer_id, account_type, balance,
                 currency, status, ifsc_code, bank_name)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (account_ref) DO NOTHING
            """,
            [
                (
                    a["id"],
                    a["account_ref"],
                    a["customer_id"],
                    a["account_type"],
                    a["balance"],
                    a["currency"],
                    a["status"],
                    a["ifsc_code"],
                    a["bank_name"],
                )
                for a in accounts
            ],
        )
        log.info("inserted_accounts", count=len(accounts))

        # ── Insert transactions ───────────────────────────────────
        import json

        await conn.executemany(
            """
            INSERT INTO banking.transactions
                (id, transaction_ref, debit_account_id, credit_account_id,
                 amount, currency, transaction_type, status, reference_number,
                 description, initiated_at, completed_at, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (transaction_ref) DO NOTHING
            """,
            [
                (
                    t["id"],
                    t["transaction_ref"],
                    t["debit_account_id"],
                    t["credit_account_id"],
                    t["amount"],
                    t["currency"],
                    t["transaction_type"],
                    t["status"],
                    t["reference_number"],
                    t["description"],
                    t["initiated_at"],
                    t["completed_at"],
                    json.dumps(t["metadata"]),
                )
                for t in transactions
            ],
        )
        log.info("inserted_transactions", count=len(transactions))

        # ── Insert payment events ─────────────────────────────────
        await conn.executemany(
            """
            INSERT INTO banking.payment_events
                (id, transaction_id, event_type, event_code, event_message,
                 payment_rail, occurred_at, source_system, raw_payload)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            [
                (
                    e["id"],
                    e["transaction_id"],
                    e["event_type"],
                    e["event_code"],
                    e["event_message"],
                    e["payment_rail"],
                    e["occurred_at"],
                    e["source_system"],
                    json.dumps(e["raw_payload"]),
                )
                for e in events
            ],
        )
        log.info("inserted_payment_events", count=len(events))

        # ── Insert policies ───────────────────────────────────────
        for p in POLICIES:
            await conn.execute(
                """
                INSERT INTO banking.policies
                    (policy_ref, title, category, content, effective_date, version, active)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (policy_ref) DO UPDATE
                    SET content = EXCLUDED.content,
                        title   = EXCLUDED.title
                """,
                p["policy_ref"],
                p["title"],
                p["category"],
                p["content"],
                p["effective_date"],
                "1.0",
                True,
            )
        log.info("inserted_policies", count=len(POLICIES))

        log.info("seeding_complete")

    finally:
        await conn.close()


if __name__ == "__main__":
    from config import settings

    # Strip asyncpg prefix for direct asyncpg connection
    url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    asyncio.run(seed(url))
