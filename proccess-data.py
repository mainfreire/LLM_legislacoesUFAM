import json
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm

file_path = 'env/documentos_divididos.json'
with open(file_path, 'r', encoding='utf-8') as file:
    dados = json.load(file)

# Inicializar o modelo e o tokenizador GPT-2
model_name = "pierreguillou/gpt2-small-portuguese"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Função para gerar perguntas e respostas
def gerar_pergunta_resposta(conteudo):
    inputs = tokenizer.encode(conteudo, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pergunta_resposta = generated_text.split('Resposta:')
    if len(pergunta_resposta) == 2:
        pergunta = pergunta_resposta[0].replace('Pergunta:', '').strip()
        resposta = pergunta_resposta[1].strip()
        return {
            "pergunta": pergunta,
            "resposta": resposta
        }
    return None

# Processar os documentos e gerar perguntas e respostas
dados_sinteticos = []
num_perguntas_por_documento = 5  # Ajuste conforme necessário

for documento in tqdm(dados, desc="Processando documentos"):
    conteudo = documento['conteudo']
    for _ in range(num_perguntas_por_documento):
        pr = gerar_pergunta_resposta(conteudo)
        if pr and len(dados_sinteticos) < 1000:  # Limitar a 1000 exemplos
            dados_sinteticos.append({
                'documento_id': documento['documento_id'],
                'documento_titulo': documento['documento_titulo'],
                'parte_id': documento['parte_id'],
                'pergunta': pr['pergunta'],
                'resposta': pr['resposta']
            })
        if len(dados_sinteticos) >= 1000:
            break
    if len(dados_sinteticos) >= 1000:
        break

random.shuffle(dados_sinteticos)

output_file_path = 'env/perguntas_respostas.json'
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(dados_sinteticos, file, ensure_ascii=False, indent=4)

print(f'Perguntas e respostas salvas em {output_file_path}')
