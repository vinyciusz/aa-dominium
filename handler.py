import os
import torch
import runpod
from unsloth import FastLlamaModel
from transformers import TextStreamer

# --- VARIÁVEIS DE CONFIGURAÇÃO ---
# O nome do modelo base e o caminho onde o modelo treinado será salvo no container
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ADAPTERS_PATH = "/workspace/usucapiao_8b_v1_lora" # O nome da pasta que você baixou

# Configurações de hardware e precisão (deve ser o mesmo que o treino)
MAX_SEQ_LENGTH = 1024 
DTYPE = torch.bfloat16
LOAD_IN_4BIT = True
DEVICE = "cuda" # Indica que usaremos a GPU

# Armazenamento global para evitar carregar o modelo em cada requisição
MODEL = None
TOKENIZER = None

# --- FUNÇÃO PRINCIPAL: Carregar o Modelo ---
def load_model():
    """Carrega o modelo base e aplica os adaptadores LoRA treinados."""
    global MODEL, TOKENIZER

    # 1. Carrega o modelo base (Llama 3 8B 4-bit)
    model, tokenizer = FastLlamaModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        device_map = {"": DEVICE},
        token = None,
    )

    # 2. Aplica os adaptadores (seu cérebro treinado)
    try:
        model.load_adapter(ADAPTERS_PATH)
        print("✅ Adaptadores LoRA aplicados!")
    except Exception as e:
        # Se o modelo ainda não estiver sido descompactado, este aviso aparece, mas o servidor continua.
        print(f"AVISO: Falha ao carregar adaptadores LoRA. Erro: {e}. O modelo rodará sem a especialização (apenas Llama 3 8B base).")
        pass 
        
    # 3. Finaliza a configuração
    model.eval()
    TOKENIZER = tokenizer
    MODEL = model
    return MODEL, TOKENIZER

# --- FUNÇÃO DE SERVIÇO: Onde o RunPod envia a requisição ---
def handler(job):
    """
    Processa a requisição do usuário (via WhatsApp/Twilio) e gera a resposta da IA.
    Esta função é chamada em cada requisição da API.
    """
    
    # 1. Garante que o modelo está carregado na memória
    if MODEL is None or TOKENIZER is None:
        load_model()
        
    # 2. Extrai os dados do job (que virão da requisição POST)
    # O input deve ser passado como {"input": {"prompt": "sua pergunta aqui"}}
    try:
        prompt = job["input"]["prompt"]
    except KeyError:
        return {"error": "JSON de entrada inválido. Requer 'prompt' em 'input'."}

    # 3. Formata o prompt no template de chat do Llama 3
    template_chat = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    prompt_formatado = template_chat.format(prompt)
    
    # 4. Tokeniza o input e move para a GPU
    inputs = TOKENIZER([prompt_formatado], return_tensors = "pt", padding = True).to(DEVICE)

    # 5. Gera a resposta
    outputs = MODEL.generate(
        **inputs, 
        max_new_tokens = 512, 
        use_cache = True,
        do_sample = True,
        temperature = 0.7, # Balanceado para precisão e criatividade
        repetition_penalty = 1.05,
        pad_token_id = TOKENIZER.eos_token_id # Garante que o modelo não continue gerando
    )
    
    # 6. Decodifica e retorna
    resposta_da_ia = TOKENIZER.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    return {"result": resposta_da_ia}

# --- INICIALIZAÇÃO ---
if __name__ == '__main__':
    # Esta linha inicia o handler do RunPod quando o container é iniciado
    runpod.serverless.start({"handler": handler})
