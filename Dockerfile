# Usar a imagem oficial do RunPod (Estável e Rápida)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório
WORKDIR /workspace

# Instala o Unsloth diretamente do GitHub (A forma mais segura hoje)
RUN pip install --upgrade pip
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Copia e instala os requisitos extras (PDF, Word, API)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia o código da IA
COPY . .

# Comando de início (Chama seu script handler.py diretamente)
CMD [ "python3", "-u", "handler.py" ]
