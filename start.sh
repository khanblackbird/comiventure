#!/bin/bash
# Comiventure server control
#
# Usage:
#   ./start.sh              Start server (no proxy)
#   ./start.sh proxy        Start server + WARP proxy for Civitai
#   ./start.sh restart      Restart the app container (with checks)
#   ./start.sh stop         Stop everything
#   ./start.sh logs         Tail the app logs
#   ./start.sh test         Run backend tests (fast, no browser)
#   ./start.sh test-ui      Run UI tests (Playwright, launches browser)
#   ./start.sh test-all     Run all tests
#   ./start.sh setup-proxy  First-time WARP proxy install

set -e

# ---------------------------------------------------------------------------
# Colors / helpers
# ---------------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[!!]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
step()  { echo -e "\n--- $* ---"; }

# ---------------------------------------------------------------------------
# Pre-build checks (no container image needed)
# ---------------------------------------------------------------------------

check_docker() {
    if ! sudo docker info &>/dev/null; then
        fail "Docker daemon is not running"
        echo "     Try: sudo systemctl start docker"
        exit 1
    fi
    info "Docker daemon running"
}

check_nvidia_driver() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found — no NVIDIA driver installed"
        warn "Image generation will be disabled"
        return 1
    fi
    if ! nvidia-smi &>/dev/null; then
        warn "nvidia-smi failed — GPU may be in a bad state"
        warn "Try: sudo nvidia-smi -r (reset) or reboot"
        return 1
    fi
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    info "NVIDIA driver OK ($gpu_name)"
    return 0
}

check_ram() {
    local total_kb
    total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    local total_gb=$((total_kb / 1024 / 1024))
    if [ "$total_gb" -lt 12 ]; then
        warn "Low RAM: ${total_gb}GB — SDXL needs 12GB+ (RAM + swap)"
    else
        info "RAM OK (${total_gb}GB)"
    fi
}

check_swap() {
    local swap_total
    swap_total=$(awk '/SwapTotal/{print $2}' /proc/meminfo)
    if [ "$swap_total" -eq 0 ] 2>/dev/null; then
        warn "No swap space — SDXL generation may OOM"
        if [ -f /swapfile ]; then
            echo "     Swap file exists but not active — activating..."
            sudo swapon /swapfile 2>/dev/null && info "Swap activated" || warn "Failed to activate /swapfile"
        else
            echo "     Creating 16G swap file..."
            sudo fallocate -l 16G /swapfile && \
            sudo chmod 600 /swapfile && \
            sudo mkswap /swapfile && \
            sudo swapon /swapfile && \
            info "16G swap activated"
            if ! grep -q '/swapfile' /etc/fstab 2>/dev/null; then
                echo '/swapfile none swap defaults 0 0' | sudo tee -a /etc/fstab > /dev/null
                echo "     Added swap to /etc/fstab"
            fi
        fi
    else
        local swap_mb=$((swap_total / 1024))
        info "Swap available (${swap_mb}M)"
    fi
}

check_disk() {
    local free_kb
    free_kb=$(df --output=avail / 2>/dev/null | tail -1 | tr -d ' ')
    if [ -z "$free_kb" ]; then
        return
    fi
    local free_gb=$((free_kb / 1024 / 1024))
    if [ "$free_gb" -lt 20 ]; then
        warn "Low disk space: ${free_gb}GB free — model downloads may fail (need ~20GB)"
    else
        info "Disk space OK (${free_gb}GB free)"
    fi
}

check_ollama_installed() {
    if command -v ollama &>/dev/null; then
        if systemctl is-active --quiet ollama 2>/dev/null; then
            info "Ollama service running"
        elif pgrep -x ollama &>/dev/null; then
            info "Ollama process running"
        else
            warn "Ollama installed but not running — try: ollama serve"
            return
        fi
        # Check required models are pulled
        local models
        models=$(ollama list 2>/dev/null | awk 'NR>1{print $1}' || true)
        for required in llama3:8b llava:7b; do
            if echo "$models" | grep -q "^${required}"; then
                info "Ollama model: $required"
            else
                warn "Ollama model '$required' not pulled — run: ollama pull $required"
            fi
        done
    else
        warn "Ollama not installed — character chat and LLM features will not work"
        echo "     Install: curl -fsSL https://ollama.com/install.sh | sh"
    fi
}

# ---------------------------------------------------------------------------
# Post-build checks (need the container image)
# ---------------------------------------------------------------------------

check_gpu() {
    # Skip if no driver on host
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        return
    fi

    # Must have the app image — caller ensures maybe_rebuild ran first
    if ! sudo docker images -q comiventure-app 2>/dev/null | grep -q .; then
        warn "No app image yet — skipping PyTorch CUDA check"
        return
    fi

    local test_cmd='python -c "import torch; assert torch.cuda.is_available(), \"no cuda\"; print(torch.cuda.get_device_name(0))"'

    local result
    result=$(sudo docker run --rm --gpus all comiventure-app \
        bash -c "$test_cmd" 2>&1)
    local rc=$?

    if [ $rc -ne 0 ]; then
        warn "PyTorch CUDA not working inside containers"
        echo "     Checking if privileged mode helps..."

        result=$(sudo docker run --rm --gpus all --privileged comiventure-app \
            bash -c "$test_cmd" 2>&1)
        rc=$?

        if [ $rc -eq 0 ]; then
            info "PyTorch CUDA works with privileged mode ($result)"
            if ! grep -q "privileged: true" docker-compose.yml; then
                echo "     Adding 'privileged: true' to docker-compose.yml"
                sed -i '/^  app:/a\    privileged: true' docker-compose.yml
            fi
        else
            fail "PyTorch CUDA fails even with privileged mode"
            echo "     Try: sudo systemctl restart docker, or reboot"
            echo "     Image generation will be disabled"
        fi
    else
        info "PyTorch CUDA OK ($result)"
    fi
}

# ---------------------------------------------------------------------------
# Build check
# ---------------------------------------------------------------------------

maybe_rebuild() {
    local image_id
    image_id=$(sudo docker images -q comiventure-app 2>/dev/null)
    if [ -z "$image_id" ]; then
        echo "No existing image — building..."
        sudo docker compose build app
        return
    fi

    local image_created
    image_created=$(sudo docker inspect -f '{{.Created}}' "$image_id" 2>/dev/null)
    image_ts=$(date -d "$image_created" +%s 2>/dev/null || echo 0)

    local needs_rebuild=false
    for f in Dockerfile requirements.txt; do
        if [ -f "$f" ]; then
            file_ts=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            if [ "$file_ts" -gt "$image_ts" ]; then
                needs_rebuild=true
                echo "Detected change: $f"
            fi
        fi
    done

    if [ "$needs_rebuild" = true ]; then
        echo "Rebuilding app image..."
        sudo docker compose build app
    else
        info "Image up to date"
    fi
}

# ---------------------------------------------------------------------------
# Git integration
# ---------------------------------------------------------------------------

check_git() {
    if ! command -v git &>/dev/null || ! git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
        return
    fi

    local branch
    branch=$(git branch --show-current 2>/dev/null)
    info "Branch: ${branch:-detached}"

    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        warn "Uncommitted changes present"
    else
        info "Working tree clean"
    fi

    # Fetch and check for upstream changes
    git fetch --quiet 2>/dev/null || true
    local behind
    behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0)
    if [ "$behind" -gt 0 ]; then
        warn "Branch is $behind commit(s) behind upstream"
        if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
            echo "     Working tree is clean — pulling..."
            git pull --ff-only 2>/dev/null && info "Pulled $behind commit(s)" \
                || warn "Fast-forward pull failed — manual merge needed"
        else
            warn "Cannot auto-pull with uncommitted changes"
        fi
    else
        info "Up to date with upstream"
    fi
}

# ---------------------------------------------------------------------------
# Health checks after startup
# ---------------------------------------------------------------------------

wait_for_server() {
    step "Waiting for server"
    local elapsed=0
    while [ $elapsed -lt 60 ]; do
        if curl -sf http://localhost:8000/ -o /dev/null 2>/dev/null; then
            info "Server responding (${elapsed}s)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "Server failed to start within 60 seconds"
    return 1
}

check_ollama_reachable() {
    if curl -sf http://localhost:11434/ -o /dev/null 2>/dev/null; then
        info "Ollama reachable on port 11434"
    else
        warn "Ollama not reachable on port 11434 — LLM features may not work"
    fi
}

verify_gpu_loaded() {
    # Check the container logs for the GPU loaded / disabled message
    local logs
    logs=$(sudo docker compose logs --tail=30 app 2>/dev/null)

    if echo "$logs" | grep -q "Image generator loaded on"; then
        local gpu_name
        gpu_name=$(echo "$logs" | grep "Image generator loaded on" | sed 's/.*loaded on //')
        info "Image generator loaded ($gpu_name)"
    elif echo "$logs" | grep -q "No CUDA GPU found"; then
        fail "Image generator FAILED — No CUDA GPU found"
        echo "     PyTorch CUDA init failed inside the running container."
        echo "     Check: docker-compose.yml has 'privileged: true'"
        echo "     Check: sudo docker compose logs app | grep -i cuda"
        echo "     Try:   sudo docker compose down && ./start.sh"
    elif echo "$logs" | grep -q "PyTorch not installed"; then
        fail "Image generator FAILED — PyTorch not installed"
    else
        warn "Could not determine image generator status — check logs"
    fi
}

# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------

ensure_clean() {
    if sudo docker compose ps --status running 2>/dev/null | grep -q "app"; then
        echo "Server already running — stopping..."
        sudo docker compose down
    fi
}

# ---------------------------------------------------------------------------
# Startup sequences
# ---------------------------------------------------------------------------

run_preflight() {
    step "System checks"
    check_docker
    check_nvidia_driver || true
    check_ram
    check_disk
    check_swap
    check_ollama_installed

    step "Git status"
    check_git

    step "Build"
    maybe_rebuild

    # GPU check AFTER build — needs the app image for PyTorch test
    step "GPU check"
    check_gpu
}

run_post_startup() {
    wait_for_server || true

    step "Connection checks"
    check_ollama_reachable

    # Wait a moment for the startup event to fire, then verify
    sleep 3
    step "Verifying image generator"
    verify_gpu_loaded
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

case "${1:-}" in
    proxy)
        ensure_clean

        echo "Starting WARP proxy..."
        warp-cli connect 2>/dev/null || echo "WARP not connected — run './start.sh setup-proxy' first"
        sleep 1

        if ss -tln | grep -q ':40000'; then
            info "WARP proxy running on localhost:40000"
            export CIVITAI_PROXY=socks5://127.0.0.1:40000
        else
            warn "WARP proxy not detected on port 40000"
            echo "     Civitai browser will not work"
        fi

        run_preflight

        step "Starting server with proxy"
        sudo CIVITAI_PROXY="$CIVITAI_PROXY" docker compose up -d

        run_post_startup

        echo ""
        echo "Server: http://localhost:8000"
        echo "Civitai: proxied through WARP"
        sudo docker compose logs -f app
        ;;

    restart)
        step "Pre-restart checks"
        check_docker
        check_nvidia_driver || true

        # Rebuild if needed before restart
        step "Build"
        maybe_rebuild

        step "Restarting"
        sudo docker compose down
        sudo docker compose up -d

        run_post_startup

        echo ""
        echo "Server: http://localhost:8000"
        sudo docker compose logs -f app
        ;;

    stop)
        echo "Stopping server..."
        sudo docker compose down
        warp-cli disconnect 2>/dev/null && echo "WARP disconnected" || true
        echo "Done"
        ;;

    logs)
        sudo docker compose logs -f app
        ;;

    test)
        echo "Running backend tests..."
        .venv/bin/python -m pytest tests/ --ignore=tests/test_ui.py -v
        ;;

    test-ui)
        echo "Running UI tests (Playwright)..."
        .venv/bin/python -m pytest tests/test_ui.py -v
        ;;

    test-all)
        echo "Running all tests..."
        echo ""
        echo "=== Backend ==="
        .venv/bin/python -m pytest tests/ --ignore=tests/test_ui.py -v
        echo ""
        echo "=== UI (Playwright) ==="
        .venv/bin/python -m pytest tests/test_ui.py -v
        ;;

    setup-proxy)
        echo "Installing Cloudflare WARP..."
        sudo pacman -S --needed base-devel git

        if ! pacman -Q cloudflare-warp-bin &>/dev/null; then
            cd /tmp
            rm -rf cloudflare-warp-bin
            git clone https://aur.archlinux.org/cloudflare-warp-bin.git
            cd cloudflare-warp-bin
            makepkg -si
            cd -
        else
            echo "cloudflare-warp-bin already installed"
        fi

        sudo systemctl enable --now warp-svc
        sleep 2
        warp-cli registration new 2>/dev/null || echo "Already registered"
        warp-cli mode proxy
        echo ""
        echo "Done. Run: ./start.sh proxy"
        ;;

    "")
        ensure_clean
        run_preflight

        step "Starting server (no proxy)"
        sudo docker compose up -d

        run_post_startup

        echo ""
        echo "Server: http://localhost:8000"
        echo "Civitai: no proxy (may be region-blocked)"
        sudo docker compose logs -f app
        ;;

    *)
        echo "Usage: ./start.sh [command]"
        echo ""
        echo "  (no args)     Start server without proxy"
        echo "  proxy         Start server + WARP proxy for Civitai"
        echo "  restart       Restart the app container (with checks)"
        echo "  stop          Stop server and disconnect proxy"
        echo "  logs          Tail the app logs"
        echo "  test          Run backend tests (fast)"
        echo "  test-ui       Run UI tests (Playwright)"
        echo "  test-all      Run all tests"
        echo "  setup-proxy   First-time WARP proxy install"
        ;;
esac
