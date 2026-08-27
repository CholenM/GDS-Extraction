# ===========================================================================
# AI GDS Extraction — Launch Script (DGX Spark)
# ===========================================================================
# The model server is ALREADY RUNNING on the DGX (now vLLM, not llama.cpp).
# This script only launches the FastAPI GATEWAY and connects to that model.
# It does NOT start, stop, or kill the model server.
#
# vLLM migration (2026-08-27): default is now vLLM Qwen3.6-35B-A3B-NVFP4 on :8011
# (managed by ~/vllm-qwen/startserver.sh or E:\DGXSpark_Setup\vllm-qwen\startserver.sh)
# Legacy llama.cpp :8006 still works if you set MODEL_URL in .env — this script
# auto-detects the port from MODEL_URL.
#
# Lightweight replacement for setup.sh: if the venv or .env don't exist yet it
# bootstraps them, so ./start.sh remains a single entry point.
#
# Usage: ./start.sh
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ---------------------------------------------------------------------------
# Step 1: Bootstrap venv (one-time) — mirrors the dropped setup.sh venv step
# ---------------------------------------------------------------------------
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${CYAN}[1/3] Creating Python virtual environment...${NC}"
    python3 -m venv "$SCRIPT_DIR/.venv"
fi
source "$SCRIPT_DIR/.venv/bin/activate"

if ! python -c "import fastapi, requests, dotenv" >/dev/null 2>&1; then
    echo -e "${CYAN}[1/3] Installing runtime dependencies...${NC}"
    pip install --upgrade pip -q
    pip install -r "$SCRIPT_DIR/requirements.txt" -q
fi

# ---------------------------------------------------------------------------
# Step 2: Bootstrap .env (one-time) from the example
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${CYAN}[2/3] .env not found — creating from .env.example${NC}"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo -e "${YELLOW}  .env generated. Review it (API keys, model URL, ports) before continuing.${NC}"
fi

# Load .env
set -a; source .env; set +a

MODEL_URL="${MODEL_URL:-http://127.0.0.1:8011/v1/chat/completions}"
# Shared model server port, parsed from MODEL_URL (the single source of truth).
# Defaults to 8011 (vLLM) if MODEL_URL has no :port. Kept defined so `set -u` is happy.
_MODEL_PORT="${MODEL_URL#*://}"; _MODEL_PORT="${_MODEL_PORT%%/*}"; _MODEL_PORT="${_MODEL_PORT##*:}"
MODEL_PORT="${_MODEL_PORT//[!0-9]/}"
MODEL_PORT="${MODEL_PORT:-8011}"
API_PORT="${API_PORT:-8084}"
API_HOST="${API_HOST:-0.0.0.0}"
# How long to wait for the model server to be healthy before launching gateway.
# vLLM first boot JIT-compiles FlashInfer (fused_moe, 36 kernels) → 3-15 min,
# plus CUDA-graph capture. 900s matches ~/vllm-qwen/startserver.sh health wait.
SERVER_WAIT="${SERVER_WAIT:-900}"
# How long to wait for OUR gateway's own /healthz.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"

# ---------------------------------------------------------------------------
# Step 3: Pre-flight — require the model server to be healthy
#         (managed externally: ~/vllm-qwen/startserver.sh for vLLM :8011,
#          or Proof-Reader/startserver.sh for legacy llama.cpp :8006)
# ---------------------------------------------------------------------------
echo -e "${CYAN}[3/3] Pre-flight: model server ${MODEL_URL} (timeout ${SERVER_WAIT}s)...${NC}"
echo -e "  MODEL_URL=$MODEL_URL  MODEL_NAME=${MODEL_NAME:-<from .env>}  CONTEXT_SIZE=${CONTEXT_SIZE:-?}"
# vLLM migration check: warn if .env still points to legacy :8006
if [[ "$MODEL_URL" == *":8006"* ]]; then
    echo -e "${YELLOW}  WARN: MODEL_URL still points to llama.cpp :8006 — for vLLM use :8011 (MODEL_URL=http://127.0.0.1:8011/v1/chat/completions, MODEL_NAME=Qwen3.6-35B-A3B-NVFP4, CONTEXT_SIZE=32768, MODEL_PARALLEL=1)${NC}"
fi
MODEL_HEALTH_URL="${MODEL_URL%/v1/chat/completions}/health"
MODEL_READY="no"
ELAPSED=0
while [ $ELAPSED -lt "$SERVER_WAIT" ]; do
    if curl -sf "$MODEL_HEALTH_URL" >/dev/null 2>&1; then
        MODEL_READY="yes"
        break
    fi
    # also probe /v1/models as vLLM fallback (some builds don't expose /health immediately)
    if curl -sf "${MODEL_URL%/v1/chat/completions}/v1/models" >/dev/null 2>&1; then
        MODEL_READY="yes"
        break
    fi
    sleep 2; ELAPSED=$((ELAPSED + 2))
    # periodic progress + last error hint
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo -e "  ... still waiting (${ELAPSED}s) — curl -v $MODEL_HEALTH_URL"
        curl -sv "$MODEL_HEALTH_URL" 2>&1 | head -n 5 || true
    fi
done
if [ "$MODEL_READY" != "yes" ]; then
    echo -e "${RED}ERROR: Model server not reachable at ${MODEL_HEALTH_URL} after ${SERVER_WAIT}s.${NC}"
    echo -e "  For vLLM (Qwen3.6-35B-A3B-NVFP4): ${CYAN}cd ~/vllm-qwen && ./startserver.sh${NC}  then ${CYAN}tail -f ~/vllm/logs/vllm-qwen-server.log${NC}"
    echo -e "  For legacy llama.cpp: ${CYAN}cd E:\\Projects\\Proof-Reader && ./startserver.sh${NC}"
    echo -e "  Check: ${CYAN}curl -v $MODEL_HEALTH_URL ; curl -s ${MODEL_URL%/v1/chat/completions}/v1/models | head${NC}"
    echo -e "  Check .env: ${CYAN}cat .env | grep -E 'MODEL_URL|MODEL_NAME|CONTEXT_SIZE|MODEL_PARALLEL|LLAMA_SERVER_API_KEY'${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Model server healthy on :${MODEL_PORT} ($MODEL_HEALTH_URL)"
# Show which backend was detected
if curl -sf "${MODEL_URL%/v1/chat/completions}/v1/models" 2>/dev/null | grep -q "Qwen3.6"; then
    echo -e "${GREEN}  Detected vLLM Qwen3.6-35B-A3B-NVFP4${NC}"
elif curl -sf "${MODEL_URL%/v1/chat/completions}/v1/models" 2>/dev/null | grep -q "Qwen3.8"; then
    echo -e "${YELLOW}  Detected llama.cpp Qwen3.8-27B (legacy)${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Launch the FastAPI gateway ONLY
# ---------------------------------------------------------------------------
echo -e "${CYAN}Starting GDS Extraction gateway on :${API_PORT}...${NC}"
python3 gds_extraction_service.py &
API_PID=$!
echo "API_PID=$API_PID" > "$SCRIPT_DIR/.pids"

# Wait for our own /healthz before declaring LIVE (model already confirmed above).
echo -e "${CYAN}Waiting for gateway /healthz (up to ${HEALTH_TIMEOUT}s)...${NC}"
ELAPSED=0
while [ $ELAPSED -lt "$HEALTH_TIMEOUT" ]; do
    curl -sf "http://127.0.0.1:${API_PORT}/healthz" >/dev/null 2>&1 && { echo -e "${GREEN}✓${NC} Gateway healthy (${ELAPSED}s)"; break; }
    kill -0 "$API_PID" 2>/dev/null || { echo -e "${RED}FastAPI exited unexpectedly!${NC}"; exit 1; }
    sleep 2; ELAPSED=$((ELAPSED + 2))
done

echo ""
echo -e "${GREEN}======== GDS Extraction is LIVE ========${NC}"
echo -e "  API:    http://0.0.0.0:${API_PORT}"
echo -e "  Docs:   http://0.0.0.0:${API_PORT}/docs"
echo -e "  Health: http://0.0.0.0:${API_PORT}/healthz"
echo -e "  Model:  $MODEL_URL (port :${MODEL_PORT}, $(if [[ "$MODEL_URL" == *":8011"* ]]; then echo "vLLM Qwen3.6-35B-A3B-NVFP4"; else echo "llama.cpp"; fi))"
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop the gateway only — the model server is left running."
echo -e "  If you get 503 on /v1/extract: ${CYAN}curl -s http://127.0.0.1:${API_PORT}/healthz | python3 -m json.tool${NC}"
echo -e "  and check gateway logs / vLLM logs: ${CYAN}tail -f ~/vllm/logs/vllm-qwen-server.log${NC}"
echo -e "${GREEN}======================================${NC}"

# Keep alive — watch the gateway
while true; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo -e "${RED}GDS Extraction gateway exited unexpectedly.${NC}"
        break
    fi
    sleep 5
done

# Cleanup (trap handles SIGINT/SIGTERM); remove pid file
rm -f "$SCRIPT_DIR/.pids" 2>/dev/null || true
echo -e "${GREEN}GDS Extraction gateway stopped.${NC}"
