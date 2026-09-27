"""
BankGuard AI — Temporal HITL workflows package.

Temporal handles everything LangGraph cannot:
  - Durable sleep (waiting hours/days for a human)
  - Process restart safety (workflow resumes after EC2 restart)
  - Reliable retry with backoff across process boundaries
  - Compensation (undo if approval expires)

Workflow:
  ApprovalWorkflow
    └── request_approval_activity   → writes to agent.approvals
    └── wait_for_decision_activity  → polls until APPROVED/REJECTED/EXPIRED
    └── execute_approved_tool_activity → calls the actual tool

Worker entry point:  python -m workflows.temporal.worker
"""
