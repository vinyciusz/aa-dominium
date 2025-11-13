# Imagem base pequena para build rápido
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/model_cache \
    MODEL_ID=unsloth/llama-3-8b-Instruct-bnb-4bit

WORKDIR /app

# Instala dependências mínimas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copia só requirements para aproveitar cache do Docker build (se existir)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copia o restante do código
COPY . .

# Copia entrypoint que baixa o modelo na primeira execução
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Marca diretório de cache de modelo como volume (será montado pelo Runpod)
VOLUME [ "/model_cache" ]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
