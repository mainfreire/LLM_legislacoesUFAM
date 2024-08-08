import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

with open('env/dados_sinteticos.json', 'r', encoding='utf-8') as file:
    dados_sinteticos = json.load(file)

# Pré-processamento
def preprocess_data(data):
    perguntas = []
    respostas = []
    for item in data:
        perguntas.append(item['pergunta'])
        respostas.append(item['resposta'])
    return Dataset.from_dict({"pergunta": perguntas, "resposta": respostas})

dataset = preprocess_data(dados_sinteticos)

# configuraçao do modelo e tokenizer
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# configurar o LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
)

model = get_peft_model(model, lora_config)
 
# tokenização
def tokenize_function(examples):
    inputs = tokenizer(examples['pergunta'], padding="max_length", truncation=True, max_length=256)
    outputs = tokenizer(examples['resposta'], padding="max_length", truncation=True, max_length=256)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./modelo_treinado")
tokenizer.save_pretrained("./modelo_treinado")
