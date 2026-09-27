#!/usr/bin/env bash
# BankGuard AI — Local development startup script
# Usage: bash scripts/start_local.sh
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[bankguard]${NC} $1"; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────

command -v docker >/dev/null 2>&1  || err "Docker not found. Install Docker Desktop."
command -v python3 >/dev/null 2>&1 || err "Python 3 not found."

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python version: $PYTHON_VERSION"

# ── .env setup ────────────────────────────────────────────────────────────────

if [ ! -f .env ]; then
    warn ".env not found. Copying from .env.local.example..."
    cp .env.local.example .env
    warn "IMPORTANT: Edit .env and set OPENAI_API_KEY (or AWS credentials for Bedrock)."
    echo ""
    read -p "Press Enter after editing .env to continue..."
fi

# ── Python deps ───────────────────────────────────────────────────────────────

log "Installing Python dependencies..."
pip install -e ".[eval]" -q
ok "Dependencies installed"

# ── Infrastructure ────────────────────────────────────────────────────────────

log "Starting infrastructure containers..."
docker-compose -f docker-compose.local.yml up -d

log "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker exec bankguard-postgres pg_isready -U bankguard -q 2>/dev/null; then
        ok "PostgreSQL ready"
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        err "PostgreSQL did not start within 60 seconds"
    fi
done

log "Waiting for Valkey..."
for i in $(seq 1 15); do
    if docker exec bankguard-valkey valkey-cli ping 2>/dev/null | grep -q PONG; then
        ok "Valkey ready"
        break
    fi
    sleep 2
done

# ── Seed data ─────────────────────────────────────────────────────────────────

log "Seeding synthetic banking data..."
if python synthetic_data/seed_data.py 2>&1 | grep -q "seeding_complete"; then
    ok "Banking data seeded (20 customers, 100 transactions, 7 policies)"
else
    warn "Seed may have already run (duplicate key errors are normal on re-run)"
fi

# ── Ingest policies into vector store ────────────────────────────────────────

log "Ingesting policies into pgvector semantic store..."
python -c "
import asyncio, asyncpg
from memory.semantic.vector_store import SemanticMemory
from config import settings

async def main():
    url = settings.database_url.replace('postgresql+asyncpg://', 'postgresql://')
    db = await asyncpg.create_pool(url, min_size=1, max_size=3)
    sm = SemanticMemory(db)
    n = await sm.ingest_all_policies()
    print(f'Ingested {n} policy chunks into semantic store')
    await db.close()

asyncio.run(main())
" 2>/dev/null || warn "Policy ingestion skipped (embedding model not yet downloaded — will work on first search)"

# ── Print service URLs ────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  BankGuard AI — Local environment ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next: open two terminals and run:"
echo ""
echo -e "  ${CYAN}Terminal 1 — FastAPI:${NC}"
echo "    uvicorn apps.api.main:app --reload --port 8000"
echo ""
echo -e "  ${CYAN}Terminal 2 — Streamlit UI:${NC}"
echo "    streamlit run apps/operations-ui/app.py"
echo ""
echo "  Service URLs:"
echo "    🖥️  Operations UI:  http://localhost:8501"
echo "    📡  FastAPI docs:   http://localhost:8000/docs"
echo "    🔍  Langfuse:       http://localhost:3000"
echo "    📊  Grafana:        http://localhost:3002  (admin/admin)"
echo "    ⏱️  Temporal UI:    http://localhost:8088  (start temporal separately)"
echo "    📈  Prometheus:     http://localhost:9090"
echo ""
echo "  Run evaluations:"
echo "    python evals/regression/run_evals.py"
echo ""
