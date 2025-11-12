# Usa uma imagem base da NVIDIA com CUDA e Python.
# É uma prática comum para modelos grandes usar imagens otimizadas para GPU.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Define variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Instala a versão mais recente do pip
RUN python3.10 -m pip install --upgrade pip

# Cria e define o diretório de trabalho
WORKDIR /workspace

# Copia o arquivo de requisitos e instala as dependências do Python.
# Isso garante que as dependências sejam instaladas antes de copiar o código,
# otimizando o cache do Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

# Define o ponto de entrada da aplicação RunPod Serverless
# O ponto de entrada padrão é o 'runpod-python'
ENTRYPOINT ["/usr/bin/python3.10", "-m", "runpod"]
