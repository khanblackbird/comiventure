#!/bin/bash
# Comiventure server control
#
# Usage:
#   ./start.sh              Start server (no proxy)
#   ./start.sh proxy        Start server + WARP proxy for Civitai
#   ./start.sh restart      Restart the app container
#   ./start.sh stop         Stop everything
#   ./start.sh logs         Tail the app logs
#   ./start.sh test         Run backend tests (fast, no browser)
#   ./start.sh test-ui      Run UI tests (Playwright, launches browser)
#   ./start.sh test-all     Run all tests
#   ./start.sh setup-proxy  First-time WARP proxy install

set -e

# Check if containers are already running and stop them
ensure_clean() {
    if sudo docker compose ps --status running 2>/dev/null | grep -q "app"; then
        echo "Server already running — stopping..."
        sudo docker compose down
    fi
}

case "${1:-}" in
    proxy)
        ensure_clean

        echo "Starting WARP proxy..."
        warp-cli connect 2>/dev/null || echo "WARP not connected — run './start.sh setup-proxy' first"
        sleep 1

        if ss -tln | grep -q ':40000'; then
            echo "WARP proxy running on localhost:40000"
            export CIVITAI_PROXY=socks5://127.0.0.1:40000
        else
            echo "Warning: WARP proxy not detected on port 40000"
            echo "Civitai browser will not work"
        fi

        echo "Starting server with proxy..."
        sudo CIVITAI_PROXY="$CIVITAI_PROXY" docker compose up -d
        echo ""
        echo "Server: http://localhost:8000"
        echo "Civitai: proxied through WARP"
        sudo docker compose logs -f app
        ;;

    restart)
        echo "Restarting app container..."
        sudo docker compose restart app
        echo "Restarted. Tailing logs..."
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

        echo "Starting server (no proxy)..."
        sudo docker compose up -d
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
        echo "  restart       Restart the app container"
        echo "  stop          Stop server and disconnect proxy"
        echo "  logs          Tail the app logs"
        echo "  test          Run backend tests (fast)"
        echo "  test-ui       Run UI tests (Playwright)"
        echo "  test-all      Run all tests"
        echo "  setup-proxy   First-time WARP proxy install"
        ;;
esac
