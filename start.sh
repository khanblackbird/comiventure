#!/bin/bash
# Comiventure server control
#
# Usage:
#   ./start.sh              Start server (no proxy)
#   ./start.sh proxy        Start server + WARP proxy for Civitai
#   ./start.sh stop         Stop everything
#   ./start.sh logs         Tail the app logs
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

    stop)
        echo "Stopping server..."
        sudo docker compose down
        warp-cli disconnect 2>/dev/null && echo "WARP disconnected" || true
        echo "Done"
        ;;

    logs)
        sudo docker compose logs -f app
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
        echo "Usage: ./start.sh [proxy|stop|logs|setup-proxy]"
        echo ""
        echo "  (no args)     Start server without proxy"
        echo "  proxy         Start server + WARP proxy for Civitai"
        echo "  stop          Stop server and disconnect proxy"
        echo "  logs          Tail the app logs"
        echo "  setup-proxy   First-time WARP proxy install"
        ;;
esac
