#!/usr/bin/env bash
set -euo pipefail

# Variáveis (podem ser setadas pelo Runpod / ambiente)
: "${TRANSFORMERS_CACHE:=/model_cache}"
: "${MODEL_ID:=unsloth/llama-3-8b-Instruct-bnb-4bit}"
: "${HF_TOKEN:=}"

echo "[entrypoint] TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"
chmod 777 "$TRANSFORMERS_CACHE" || true

# Verifica se já há arquivos no cache (se sim, pula o download)
if [ -n "$(ls -A "$TRANSFORMERS_CACHE" 2>/dev/null || true)" ]; then
  echo "[entrypoint] Cache detectado em $TRANSFORMERS_CACHE — pulando download."
else
  echo "[entrypoint] Nenhum cache encontrado. Iniciando download do modelo $MODEL_ID ..."
  python - <<PY
import os
from huggingface_hub import snapshot_download
cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/model_cache")
model_id = os.environ.get("MODEL_ID", "unsloth/llama-3-8b-Instruct-bnb-4bit")
token = os.environ.get("HF_TOKEN") or None
try:
    snapshot_download(repo_id=model_id, cache_dir=cache_dir, use_auth_token=token, force_download=False)
    print("[entrypoint][python] Download concluído.")
except Exception as e:
    print("[entrypoint][python] Erro ao baixar o modelo:", e)
    raise
PY
fi

# Executa o comando principal do container
exec "$@"
