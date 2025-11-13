# Usa a imagem oficial do RunPod (que já tem PyTorch e CUDA)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório
WORKDIR /workspace

# Copia seus arquivos
COPY . .

# 1. Instala as ferramentas básicas (Rápido)
RUN pip install --upgrade pip
RUN pip install runpod fastapi uvicorn python-docx PyPDF2 protobuf scipy

# 2. Instala o Unsloth SEM baixar dependências pesadas (O Pulo do Gato)
# Usamos --no-deps para impedir que ele baixe os 15GB do PyTorch de novo
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps

# 3. Instala apenas as dependências leves que o Unsloth precisa
RUN pip install "peft" "accelerate" "bitsandbytes" "trl" "transformers"

# Comando de início
CMD [ "python3", "-u", "handler.py" ]
