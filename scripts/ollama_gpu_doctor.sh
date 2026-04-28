#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
OLLAMA_HOST_URL="${OLLAMA_HOST:-http://127.0.0.1:11435}"

print_header() {
  printf '\n== %s ==\n' "$1"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/ollama_gpu_doctor.sh

This script only checks the project Docker Ollama endpoint from the host.
It does not modify system files, Docker settings, drivers, or Ollama data.

Environment overrides:
  OLLAMA_MODEL=qwen2.5:7b
  OLLAMA_HOST=http://127.0.0.1:11435
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

print_header "Expected Docker Path"
cat <<EOF
Feishu -> rag-feishu-bot -> rag-api -> ollama:11434
Host probe endpoint: ${OLLAMA_HOST_URL}
Expected model: ${MODEL}
Expected processor after a request: GPU or partial GPU, not 100% CPU
EOF

print_header "Host GPU Visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "nvidia-smi not found on host."
fi

print_header "Docker Ollama Models"
curl --noproxy '*' -sS --max-time 10 "${OLLAMA_HOST_URL}/api/tags" || true
echo

print_header "Tiny Generation Probe"
curl --noproxy '*' -sS --max-time 120 "${OLLAMA_HOST_URL}/api/generate" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"只回答两个字：你好\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":16,\"num_gpu\":-1}}" \
  || true
echo

print_header "Manual GPU Verification Commands"
cat <<EOF
Run these yourself if your user needs sudo for Docker:

sudo docker exec rag-ollama ollama ps
sudo docker exec rag-ollama nvidia-smi

If docker compose was already running before the config change:

docker compose up -d --force-recreate api ollama
docker compose --profile feishu up -d --force-recreate feishu-bot
EOF
