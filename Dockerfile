# --- CORREÇÃO: Usar a imagem base do Unsloth ---
# Ela já tem PyTorch, CUDA e Unsloth instalados.
# Isso reduz o tempo de build de 40min para 2min.
FROM unslothai/unsloth:latest

# Define o diretório de trabalho
WORKDIR /workspace

# Copia seus arquivos
COPY . .

# Instala apenas as bibliotecas EXTRAS (o básico já vem na imagem)
# O pip vai verificar o requirements.txt e pular o que já tem.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Comando de inicialização
ENTRYPOINT ["python3", "-m", "runpod.serverless.start", "--handler", "handler.handler"]
