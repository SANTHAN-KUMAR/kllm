#!/usr/bin/env bash
# setup_named_tunnel.sh — Set up a named cloudflared tunnel for stable 2-4 hour sessions
# Run this on your LOCAL machine (not in Kaggle)
set -euo pipefail

TUNNEL_NAME="${1:-kllm}"
HOSTNAME="${2:-}"
KAGGLE_USER="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username (owner of the credentials dataset)}"

echo "=== Named Cloudflared Tunnel Setup ==="
echo ""

# Check cloudflared
CF=""
for c in "./bin/cloudflared" "/usr/local/bin/cloudflared" "$(command -v cloudflared 2>/dev/null)"; do
    if [ -x "$c" ]; then CF="$c"; break; fi
done
if [ -z "$CF" ]; then
    echo "ERROR: cloudflared not found. Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
    exit 1
fi
echo "cloudflared: $CF"

# Step 1: Login
if [ ! -f "$HOME/.cloudflared/cert.pem" ]; then
    echo ""
    echo "Step 1: Login to Cloudflare (opens browser)..."
    "$CF" tunnel login
else
    echo "Step 1: Already logged in (cert.pem exists)"
fi

# Step 2: Create tunnel
echo ""
echo "Step 2: Creating tunnel '$TUNNEL_NAME'..."
"$CF" tunnel create "$TUNNEL_NAME" 2>&1 || echo "(tunnel may already exist)"

# Find credentials file
CREDS=$(ls "$HOME/.cloudflared"/*.json 2>/dev/null | head -1)
if [ -z "$CREDS" ]; then
    echo "ERROR: No credentials file found in ~/.cloudflared/"
    exit 1
fi
echo "Credentials: $CREDS"

# Step 3: Route DNS (optional)
if [ -n "$HOSTNAME" ]; then
    echo ""
    echo "Step 3: Routing DNS for $HOSTNAME..."
    "$CF" tunnel route dns "$TUNNEL_NAME" "$HOSTNAME"
fi

# Step 4: Prepare Kaggle dataset
echo ""
echo "Step 4: Preparing Kaggle dataset..."
DATASET_DIR="/tmp/kllm-tunnel-dataset"
mkdir -p "$DATASET_DIR"
cp "$CREDS" "$DATASET_DIR/credentials.json"
cat > "$DATASET_DIR/dataset-metadata.json" <<EOF
{
  "title": "kllm-tunnel-credentials",
  "id": "${KAGGLE_USER}/kllm-tunnel-credentials",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
cat > "$DATASET_DIR/README.md" <<'EOF'
# KLLM Tunnel Credentials
Place your cloudflared tunnel credentials file here as `credentials.json`.
The kernel will automatically find and use it for the named tunnel.
EOF
echo "Dataset prepared at: $DATASET_DIR"

# Step 5: Upload to Kaggle
echo ""
echo "Step 5: Upload to Kaggle..."
if command -v kaggle >/dev/null 2>&1; then
    cd "$DATASET_DIR"
    kaggle datasets create -p . 2>&1 || kaggle datasets version -p . -m "update credentials" 2>&1
    echo ""
    echo "Dataset uploaded. Add '${KAGGLE_USER}/kllm-tunnel-credentials' to your kernel metadata."
else
    echo "kaggle CLI not found. Upload manually:"
    echo "  1. Go to https://www.kaggle.com/datasets"
    echo "  2. Create new dataset: 'kllm-tunnel-credentials'"
    echo "  3. Upload: $DATASET_DIR/credentials.json"
fi

# Step 6: Update config.yaml
echo ""
echo "Step 6: Update config.yaml with:"
echo "  tunnel_name: $TUNNEL_NAME"
echo "  tunnel_credentials_file: $CREDS"
if [ -n "$HOSTNAME" ]; then
    echo "  tunnel_hostname: https://$HOSTNAME"
fi

echo ""
echo "=== Setup complete ==="
echo "The kernel will automatically use the named tunnel when credentials.json is found in a dataset."