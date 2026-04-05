#!/bin/bash
# Comiventure startup — checks, fixes, tests, starts.
# Just run ./start.sh — it handles everything.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
step() { echo -e "\n═══ $* ═══"; }

# ---------------------------------------------------------------------------
step "System"
# ---------------------------------------------------------------------------

# Docker
if ! sudo docker info &>/dev/null; then
    warn "Docker not running — starting..."
    sudo systemctl start docker
    sleep 2
    if ! sudo docker info &>/dev/null; then
        fail "Cannot start Docker"; exit 1
    fi
fi
ok "Docker"

# GPU driver
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $gpu_name"
else
    warn "No NVIDIA GPU — image generation disabled"
fi

# RAM
total_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$total_gb" -lt 12 ]; then
    warn "RAM: ${total_gb}GB (SDXL needs 12GB+ with swap)"
else
    ok "RAM: ${total_gb}GB"
fi

# Swap
swap_total=$(awk '/SwapTotal/{print $2}' /proc/meminfo)
if [ "$swap_total" -eq 0 ] 2>/dev/null; then
    warn "No swap — creating 16G..."
    if [ -f /swapfile ]; then
        sudo swapon /swapfile 2>/dev/null && ok "Swap activated" || warn "Failed to activate /swapfile"
    else
        sudo fallocate -l 16G /swapfile && \
        sudo chmod 600 /swapfile && \
        sudo mkswap /swapfile && \
        sudo swapon /swapfile && \
        ok "16G swap created"
        grep -q '/swapfile' /etc/fstab 2>/dev/null || \
            echo '/swapfile none swap defaults 0 0' | sudo tee -a /etc/fstab > /dev/null
    fi
else
    ok "Swap: $((swap_total / 1024))M"
fi

# Disk
free_gb=$(df --output=avail / 2>/dev/null | tail -1 | awk '{printf "%d", $1/1024/1024}')
if [ -n "$free_gb" ]; then
    if [ "$free_gb" -lt 20 ]; then
        warn "Disk: ${free_gb}GB free (models need ~20GB)"
    else
        ok "Disk: ${free_gb}GB free"
    fi
fi

# ---------------------------------------------------------------------------
step "Git"
# ---------------------------------------------------------------------------

if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    branch=$(git branch --show-current 2>/dev/null)
    ok "Branch: ${branch:-detached}"

    # Auto-pull if clean
    git fetch --quiet 2>/dev/null || true
    behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0)
    if [ "$behind" -gt 0 ]; then
        if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
            git pull --ff-only 2>/dev/null && ok "Pulled $behind commit(s)" \
                || warn "Pull failed — manual merge needed"
        else
            warn "$behind commit(s) behind, but uncommitted changes — skipping pull"
        fi
    else
        ok "Up to date"
    fi
fi

# ---------------------------------------------------------------------------
step "Docker image"
# ---------------------------------------------------------------------------

# Ensure privileged mode for CUDA (driver 590 + toolkit 1.19 requires it)
if command -v nvidia-smi &>/dev/null; then
    if ! grep -q "privileged: true" docker-compose.yml; then
        warn "Adding privileged: true to docker-compose.yml (needed for CUDA)"
        sed -i '/^  app:/a\    privileged: true' docker-compose.yml
    fi
    ok "privileged: true set"
fi

# Rebuild if Dockerfile or requirements changed
image_id=$(sudo docker images -q comiventure-app 2>/dev/null)
needs_build=false

if [ -z "$image_id" ]; then
    needs_build=true
    echo "No image — building..."
else
    image_ts=$(sudo docker inspect -f '{{.Created}}' "$image_id" 2>/dev/null)
    image_epoch=$(date -d "$image_ts" +%s 2>/dev/null || echo 0)
    for f in Dockerfile requirements.txt; do
        if [ -f "$f" ]; then
            file_epoch=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            if [ "$file_epoch" -gt "$image_epoch" ]; then
                needs_build=true
                echo "Changed: $f"
            fi
        fi
    done
fi

if [ "$needs_build" = true ]; then
    sudo docker compose build app
    ok "Image built"
else
    ok "Image up to date"
fi

# Pull ollama image if not cached
if ! sudo docker image inspect ollama/ollama &>/dev/null; then
    echo "Pulling ollama image..."
    sudo docker pull ollama/ollama
fi
ok "Ollama image ready"

# ---------------------------------------------------------------------------
step "Tests"
# ---------------------------------------------------------------------------

if [ -d .venv ] && [ -f .venv/bin/python ]; then
    if .venv/bin/python -m pytest tests/ --ignore=tests/test_ui.py -q 2>&1 | tail -5; then
        ok "Tests passed"
    else
        warn "Some tests failed — starting anyway"
    fi
else
    warn "No .venv — skipping tests"
fi

# ---------------------------------------------------------------------------
step "Starting"
# ---------------------------------------------------------------------------

# Stop anything running
if sudo docker compose ps --status running 2>/dev/null | grep -q "app\|ollama"; then
    echo "Stopping existing containers..."
    sudo docker compose down
fi

sudo docker compose up -d
ok "Containers started"

# Wait for server
echo -n "Waiting for server"
elapsed=0
while [ $elapsed -lt 60 ]; do
    if curl -sf http://localhost:8000/ -o /dev/null 2>/dev/null; then
        echo ""
        ok "Server ready (${elapsed}s)"
        break
    fi
    echo -n "."
    sleep 1
    elapsed=$((elapsed + 1))
done
if [ $elapsed -ge 60 ]; then
    echo ""
    fail "Server not responding after 60s"
fi

# ---------------------------------------------------------------------------
step "Connections"
# ---------------------------------------------------------------------------

# Ollama
if curl -sf http://localhost:11434/ -o /dev/null 2>/dev/null; then
    ok "Ollama responding"

    # Check models
    models=$(curl -sf http://localhost:11434/api/tags 2>/dev/null || true)
    for model in llama3:8b llava:7b; do
        if echo "$models" | grep -q "\"$model\""; then
            ok "Model: $model"
        else
            warn "Model '$model' missing — pulling..."
            sudo docker compose exec -T ollama ollama pull "$model" &
            echo "     (pulling in background)"
        fi
    done
else
    warn "Ollama not responding on :11434"
    echo "     Check: sudo docker compose logs ollama"
fi

# GPU verification — check container logs
sleep 3
logs=$(sudo docker compose logs --tail=30 app 2>/dev/null)
if echo "$logs" | grep -q "Image generator loaded on"; then
    gpu=$(echo "$logs" | grep "Image generator loaded on" | sed 's/.*loaded on //')
    ok "Image generator: $gpu"
elif echo "$logs" | grep -q "No CUDA GPU found"; then
    fail "Image generator FAILED — no CUDA"
    echo "     Check: sudo docker compose logs app | grep -i cuda"
else
    warn "Image generator status unknown — check logs"
fi

# ---------------------------------------------------------------------------
step "Ready"
# ---------------------------------------------------------------------------

echo ""
echo "  Server:  http://localhost:8000"
echo "  Logs:    sudo docker compose logs -f app"
echo "  Stop:    sudo docker compose down"
echo ""

sudo docker compose logs -f app
